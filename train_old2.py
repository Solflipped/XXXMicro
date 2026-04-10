import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dataset import load_uni_features, load_multi_features, load_all_uni_features, load_all_multi_features
from model.FT_transformer import FTTransformer
from model.MBT import MBT
from model.MDL4Microbiome import MDL4MIndividual, MDL4MShared
import numpy as np
import feature_selection_old2 as fs



def train(disease, feature, model_type, **params):
    """
    训练 FTTransformer 或 MBT 模型，根据 model_type 选择。

    Args:
        disease: 数据集的疾病类型
        feature: 特征类型（FT_transformer: 'ko' 或 'species'；MBT: 'ko,species'）
        model_type: 模型类型（'FT_transformer' 或 'MBT'）
        params: 模型参数（如 batch_size, learning_rate, n_blocks, fusion_layer, num_bottleneck）
    """
    # 准备 Cross-Validation 参数
    cv_splits =  5  # number of folds
    cv_repeats = 1  # number of repeats
    fs_method = 'none'  # methods supported by feature_selection.py，include: none、 all、 RandomForest+RFECV
    batch_size = params.get('batch_size', 8)

    # 加载全部样本（供外部 CV 划分使用）
    if model_type == "FT_transformer":
        if feature not in ("ko", "species"):
            raise ValueError("The feature of FT_transformer can only be 'ko' or 'species'")
        X_all, y_all, feature_list = load_all_uni_features(disease, feature)
        is_multimodal = False
    elif model_type == "MBT" or model_type == "MDL4Microbiome":
        # MBT 与 MDL4Microbiome 均强制使用多模态特征
        if ',' not in feature:
            raise ValueError("MBT/MDL4Microbiome require multimodal features (e.g., 'ko,species').")
        modality_order = feature.split(',')
        if set(modality_order) != {'ko', 'species'}:
            raise ValueError("The feature of MBT/MDL4Microbiome must be 'ko,species' or 'species,ko'")

        species_all, ko_all, y_all, species_feat_names, ko_feat_names = load_all_multi_features(disease, feature)
        X_ref = species_all
        is_multimodal = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # device: 在 CV 开始前准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Repeated Stratified K-Fold (correct for classification)
    rskf = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)

    fold_results = []
    fold_idx = 0
    total_folds = cv_splits * cv_repeats
    print(f"Starting Repeated Stratified K-Fold: {cv_splits} splits x {cv_repeats} repeats = {total_folds} folds")

    # 选择用于划分的 X（单模态用 X_all，多模态用 X_ref）
    X_for_split = X_all if not is_multimodal else X_ref

    for train_idx, val_idx in rskf.split(X_for_split, y_all):
        fold_idx += 1
        print(f"Fold {fold_idx}/{total_folds}")

        # 划分 fold 内的训练/验证数据
        if not is_multimodal:
            X_train = X_all[train_idx]
            X_val = X_all[val_idx]
        else:
            species_train = species_all[train_idx]
            species_val = species_all[val_idx]
            ko_train = ko_all[train_idx]
            ko_val = ko_all[val_idx]

        y_train_fold = np.array(y_all)[train_idx]
        y_val_fold = np.array(y_all)[val_idx]

        # 标准化（只用训练集拟合）
        if not is_multimodal:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        else:
            scaler_sp = StandardScaler()
            scaler_ko = StandardScaler()
            species_train = scaler_sp.fit_transform(species_train)
            species_val = scaler_sp.transform(species_val)
            ko_train = scaler_ko.fit_transform(ko_train)
            ko_val = scaler_ko.transform(ko_val)

        # 特征选择 —— 只在训练集上拟合选择器，按模态分别处理
        if not is_multimodal:
            # simplified call: transfer_feature_selection(train_X, train_y, test_X, feature_list, fs_method=...)
            X_train, X_val, sel_features = fs.transfer_feature_selection(
                X_train, y_train_fold, X_val, feature_list,
                fs_method=fs_method,
                n_estimators=params.get('fs_n_estimators', 200),
                rfecv_n_splits=params.get('fs_rfecv_splits', 5),
                rfecv_n_repeats=params.get('fs_rfecv_repeats', 1),
                random_state=42
            )
            # keep original feature_list intact across folds; store selected names locally
            feature_list_selected = list(sel_features) if sel_features is not None else feature_list
        else:
            sp_train_sel, sp_val_sel, sp_feats = fs.transfer_feature_selection(
                species_train, y_train_fold, species_val, species_feat_names,
                fs_method=fs_method,
                n_estimators=params.get('fs_n_estimators', 200),
                rfecv_n_splits=params.get('fs_rfecv_splits', 5),
                rfecv_n_repeats=params.get('fs_rfecv_repeats', 1),
                random_state=42
            )
            ko_train_sel, ko_val_sel, ko_feats = fs.transfer_feature_selection(
                ko_train, y_train_fold, ko_val, ko_feat_names,
                fs_method=fs_method,
                n_estimators=params.get('fs_n_estimators', 200),
                rfecv_n_splits=params.get('fs_rfecv_splits', 5),
                rfecv_n_repeats=params.get('fs_rfecv_repeats', 1),
                random_state=42
            )
            species_train = sp_train_sel
            species_val = sp_val_sel
            ko_train = ko_train_sel
            ko_val = ko_val_sel
            # do not overwrite the original feature-name lists used to index the raw arrays across folds
            species_feat_names_selected = list(sp_feats) if sp_feats is not None else species_feat_names
            ko_feat_names_selected = list(ko_feats) if ko_feats is not None else ko_feat_names
        
        # ============== MDL4Microbiome 两阶段训练路径 ==============
        if model_type == "MDL4Microbiome":
            epoch1 = int(params.get('epoch1', 30))
            epoch2 = int(params.get('epoch2', 10))

            # 准备标签张量
            ytr_t = torch.tensor(y_train_fold, dtype=torch.float32).reshape(-1, 1)
            yval_t = torch.tensor(y_val_fold, dtype=torch.float32).reshape(-1, 1)

            # Stage-1: 每模态独立训练并导出 50 维表示
            if not is_multimodal:
                # 保护性校验：按设计 MDL4Microbiome 仅支持多模态
                raise ValueError("MDL4Microbiome requires multimodal features (e.g., 'ko,species').")
            else:
                # 多模态：分别训练 species 与 ko 的 individual 模型
                sp_tr = torch.tensor(species_train, dtype=torch.float32)
                ko_tr = torch.tensor(ko_train, dtype=torch.float32)
                sp_val = torch.tensor(species_val, dtype=torch.float32)
                ko_val = torch.tensor(ko_val, dtype=torch.float32)
                ytr_t = torch.tensor(y_train_fold, dtype=torch.float32).reshape(-1, 1)
                yval_t = torch.tensor(y_val_fold, dtype=torch.float32).reshape(-1, 1)

                def train_individual(xtr, xval, in_dim, tag):
                    model_i = MDL4MIndividual.make_default(n_num_features=in_dim)
                    model_i.to(device)
                    try:
                        opt_i = model_i.make_default_optimizer()
                        for g in opt_i.param_groups:
                            g['lr'] = params.get('learning_rate', 1e-4)
                    except Exception:
                        opt_i = AdamW(model_i.parameters(), lr=params.get('learning_rate', 1e-4), weight_decay=1e-5)
                    crit_i = nn.BCEWithLogitsLoss()
                    tr_loader_i = DataLoader(TensorDataset(xtr, ytr_t), batch_size=batch_size, shuffle=True)
                    val_loader_i = DataLoader(TensorDataset(xval, yval_t), batch_size=batch_size, shuffle=False)
                    for ep in range(1, epoch1 + 1):
                        model_i.train(); loss_sum = 0.0
                        for xb, yb in tr_loader_i:
                            xb = xb.to(device); yb = yb.to(device)
                            logit = model_i(xb)
                            loss = crit_i(logit, yb)
                            opt_i.zero_grad(); loss.backward(); opt_i.step()
                            loss_sum += loss.item()
                        loss_sum /= max(1, len(tr_loader_i))
                        model_i.eval(); probs=[]; preds=[]; labels=[]
                        with torch.no_grad():
                            for xb, yb in val_loader_i:
                                p = torch.sigmoid(model_i(xb.to(device))).cpu().numpy().ravel()
                                yhat = (p > 0.5).astype(int)
                                labs = yb.numpy().ravel()
                                probs.extend(p.tolist()); preds.extend(yhat.tolist()); labels.extend(labs.tolist())
                        acc = accuracy_score(labels, preds)
                        try:
                            auc_i = roc_auc_score(labels, probs)
                        except Exception:
                            auc_i = float('nan')
                        prec = precision_score(labels, preds, zero_division=0)
                        rec = recall_score(labels, preds, zero_division=0)
                        f1 = f1_score(labels, preds, zero_division=0)
                        print(f"Fold {fold_idx} [Indiv {tag}] Epoch {ep}: Loss = {loss_sum:.4f}, ACC = {acc:.4f}, AUC = {auc_i:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}")
                    model_i.eval()
                    with torch.no_grad():
                        Z_tr_i = model_i.encode(xtr.to(device)).cpu()
                        Z_val_i = model_i.encode(xval.to(device)).cpu()
                    return model_i, Z_tr_i, Z_val_i

                sp_model, Z_sp_tr, Z_sp_val = train_individual(sp_tr, sp_val, species_train.shape[1], "species")
                ko_model, Z_ko_tr, Z_ko_val = train_individual(ko_tr, ko_val, ko_train.shape[1], "ko")

                # Stage-2 共享头
                Z_tr = torch.cat([Z_sp_tr, Z_ko_tr], dim=1)
                Z_val = torch.cat([Z_sp_val, Z_ko_val], dim=1)
                shared = MDL4MShared.make_default(concat_dim=Z_tr.shape[1])
                shared.to(device)
                try:
                    opt_sh = shared.make_default_optimizer()
                    for g in opt_sh.param_groups:
                        g['lr'] = params.get('learning_rate', 1e-4)
                except Exception:
                    opt_sh = AdamW(shared.parameters(), lr=params.get('learning_rate', 1e-4), weight_decay=1e-5)
                crit_sh = nn.BCEWithLogitsLoss()
                tr_loader_sh = DataLoader(TensorDataset(Z_tr, ytr_t), batch_size=batch_size, shuffle=True)
                val_loader_sh = DataLoader(TensorDataset(Z_val, yval_t), batch_size=batch_size, shuffle=False)

                best_auc = -np.inf
                best_precision = -np.inf
                best_recall = -np.inf
                best_probs = None
                best_labels = None
                for ep in range(1, epoch2 + 1):
                    shared.train(); loss_sum = 0.0
                    for zb, yb in tr_loader_sh:
                        zb = zb.to(device); yb = yb.to(device)
                        logit = shared(zb)
                        loss = crit_sh(logit, yb)
                        opt_sh.zero_grad(); loss.backward(); opt_sh.step()
                        loss_sum += loss.item()
                    loss_sum /= max(1, len(tr_loader_sh))

                    shared.eval(); probs=[]; preds=[]; labels=[]
                    with torch.no_grad():
                        for zb, yb in val_loader_sh:
                            p = torch.sigmoid(shared(zb.to(device))).cpu().numpy().ravel()
                            yhat = (p > 0.5).astype(int)
                            labs = yb.numpy().ravel()
                            probs.extend(p.tolist()); preds.extend(yhat.tolist()); labels.extend(labs.tolist())
                    acc = accuracy_score(labels, preds)
                    try:
                        auc = roc_auc_score(labels, probs)
                    except Exception:
                        auc = float('nan')
                    prec = precision_score(labels, preds, zero_division=0)
                    rec = recall_score(labels, preds, zero_division=0)
                    f1 = f1_score(labels, preds, zero_division=0)
                    print(f"Fold {fold_idx} [Shared] Epoch {ep}: Loss = {loss_sum:.4f}, ACC = {acc:.4f}, AUC = {auc:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}")
                    if not np.isnan(auc) and auc > best_auc:
                        best_auc = auc
                        best_precision = prec
                        best_recall = rec
                        best_probs = np.array(probs)
                        best_labels = np.array(labels)

                fold_entry = {
                    'fold': fold_idx,
                    'best_auc': best_auc,
                    'best_precision': best_precision,
                    'best_recall': best_recall,
                }
                if best_probs is not None and best_labels is not None:
                    try:
                        fpr, tpr, _ = roc_curve(best_labels, best_probs)
                        fold_entry.update({'fpr': fpr, 'tpr': tpr})
                    except Exception:
                        pass
                fold_results.append(fold_entry)

                # 清理
                del sp_model; del ko_model; del shared
                torch.cuda.empty_cache()
                continue  # 跳过后续 FT/MBT 路径
     

        # 构建 DataLoader
        if not is_multimodal:
            xtr_t = torch.tensor(X_train, dtype=torch.float32)
            xval_t = torch.tensor(X_val, dtype=torch.float32)
            ytr_t = torch.tensor(y_train_fold, dtype=torch.float32).reshape(-1, 1)
            yval_t = torch.tensor(y_val_fold, dtype=torch.float32).reshape(-1, 1)

            train_loader = DataLoader(TensorDataset(xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(xval_t, yval_t), batch_size=batch_size, shuffle=False)
        else:
            sp_tr_t = torch.tensor(species_train, dtype=torch.float32)
            ko_tr_t = torch.tensor(ko_train, dtype=torch.float32)
            sp_val_t = torch.tensor(species_val, dtype=torch.float32)
            ko_val_t = torch.tensor(ko_val, dtype=torch.float32)
            ytr_t = torch.tensor(y_train_fold, dtype=torch.float32).reshape(-1, 1)
            yval_t = torch.tensor(y_val_fold, dtype=torch.float32).reshape(-1, 1)

            train_loader = DataLoader(TensorDataset(sp_tr_t, ko_tr_t, ytr_t), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(sp_val_t, ko_val_t, yval_t), batch_size=batch_size, shuffle=False)

        # 每个 fold 新建模型和优化器
        if not is_multimodal:
            n_num_features = X_train.shape[1]
            model = FTTransformer.make_default(n_num_features=n_num_features, cat_cardinalities=None, n_blocks=params.get('n_blocks', 3), d_out=1)
        else:
            model = MBT.make_default(n_species_features=species_train.shape[1], n_ko_features=ko_train.shape[1], num_layers=params.get('n_blocks', 3), num_heads=params.get('num_heads', 8), fusion_layer=params.get('fusion_layer', None), n_bottlenecks=params.get('num_bottleneck', 4), test_with_bottlenecks=True)

        model.to(device)
        # 新的优化器
        learning_rate = params.get('learning_rate', 1e-4)
        try:
            optimizer = model.make_default_optimizer()
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        except Exception:
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        criterion = nn.BCEWithLogitsLoss()

        # fold 内训练（n_epochs）并记录最佳 val AUC 与对应预测（用于绘制 per-fold ROC）
        best_auc = -np.inf
        best_precision = -np.inf
        best_recall = -np.inf
        best_state = None
        best_probs = None
        best_labels = None
        

        n_epochs = int(params.get('n_epochs', 50))
        for epoch in range(1, n_epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                if not is_multimodal:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = model(x_batch)
                else:
                    x1_batch, x2_batch, y_batch = batch
                    raw_x = {'species': x1_batch.to(device), 'ko': x2_batch.to(device)}
                    outputs = model(raw_x)

                loss = criterion(outputs, y_batch.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / max(1, len(train_loader))

            # 验证
            model.eval()
            all_probs = []
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    if not is_multimodal:
                        x_batch, y_batch = batch
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        outputs = model(x_batch)
                    else:
                        x1_batch, x2_batch, y_batch = batch
                        raw_x = {'species': x1_batch.to(device), 'ko': x2_batch.to(device)}
                        outputs = model(raw_x)

                    probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                    preds = (probs > 0.5).astype(int)
                    labels_np = y_batch.cpu().numpy().ravel()

                    all_probs.extend(probs.tolist())
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels_np.tolist())

            # 计算指标
            acc = accuracy_score(all_labels, all_preds)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except Exception:
                auc = float('nan')
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f"Fold {fold_idx} Epoch {epoch}: Loss = {train_loss:.4f}, ACC = {acc:.4f}, AUC = {auc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

            # best model tracking (no early stopping)
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_probs = np.array(all_probs)
                best_labels = np.array(all_labels)
                # 将 Precision/Recall 固定为 best_auc 所在 epoch 的值
                best_precision = precision
                best_recall = recall


        # fold 结束，保存结果 (包含 per-fold ROC data)
        fold_entry = {
            'fold': fold_idx,
            'best_auc': best_auc,
            # 与 best_auc 同一 epoch 的 Precision/Recall
            'best_precision': best_precision,
            'best_recall': best_recall,
        }
        if best_probs is not None and best_labels is not None:
            try:
                fpr, tpr, _ = roc_curve(best_labels, best_probs)
                fold_entry.update({'fpr': fpr, 'tpr': tpr})
            except Exception:
                pass
        fold_results.append(fold_entry)
        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 汇总 CV 结果
    aucs = [f['best_auc'] for f in fold_results if not np.isnan(f['best_auc'])]
    precisions = [f.get('best_precision', float('nan')) for f in fold_results if not np.isnan(f.get('best_precision', float('nan')))]
    recalls = [f.get('best_recall', float('nan')) for f in fold_results if not np.isnan(f.get('best_recall', float('nan')))]

    mean_auc = np.mean(aucs) if len(aucs) > 0 else float('nan')
    std_auc = np.std(aucs) if len(aucs) > 0 else float('nan')
    mean_precision = np.mean(precisions) if len(precisions) > 0 else float('nan')
    std_precision = np.std(precisions) if len(precisions) > 0 else float('nan')
    mean_recall = np.mean(recalls) if len(recalls) > 0 else float('nan')
    std_recall = np.std(recalls) if len(recalls) > 0 else float('nan')

    print(
        "CV results: mean AUC = {:.4f}, std_AUC = {:.4f}, "
        "mean Precision = {:.4f}, std_Precision = {:.4f}, "
        "mean Recall = {:.4f}, std_Recall = {:.4f}".format(
            mean_auc, std_auc, mean_precision, std_precision, mean_recall, std_recall
        )
    )

    # 聚合并绘制平均 ROC（若有 per-fold ROC）
    rocs = [f for f in fold_results if 'fpr' in f and 'tpr' in f]
    if len(rocs) > 0:
        # Interpolate TPRs on a common FPR grid
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []
        for r in rocs:
            interp_tpr = np.interp(mean_fpr, r['fpr'], r['tpr'])
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        mean_tpr[-1] = 1.0

        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {np.nanmean(aucs):.4f})')
        plt.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), color='blue', alpha=0.2, label='±1 std')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean ROC across folds')
        plt.legend(loc='lower right')
        plt.savefig('mean_roc_cv.png')
        plt.close()
        print("Saved aggregated ROC as 'mean_roc_cv.png'")

    return fold_results

