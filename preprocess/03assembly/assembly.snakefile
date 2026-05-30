from os.path import join, abspath, expanduser
import sys
import os

################################################################################
# 指定项目目录 specify project directories
PROJECT_DIR = config["output_directory"]
# 将PROJECT_DIR转换成绝对路径 convert PROJECT_DIR to absolute path
if PROJECT_DIR[0] == '~':
    PROJECT_DIR = expanduser(PROJECT_DIR)
PROJECT_DIR = abspath(PROJECT_DIR)

# 读取样本表（assembly_input.txt）
sample_dict = {}
with open(config["sample_table"]) as inf:
    for line in inf:
        line = line.strip()
        if not line or line.startswith('#'):   # 跳过空行和注释/标题行
            continue

        parts = line.split('\t')
        sample = parts[0]

        # 根据列数动态解析
        r1 = parts[1] if len(parts) > 1 and parts[1] else None
        r2 = parts[2] if len(parts) > 2 and parts[2] else None
        orphans = parts[3] if len(parts) > 3 and parts[3] else None

        # 收集有效的 reads（去掉 None）
        reads = [x for x in (r1, r2, orphans) if x]

        if not reads:
            raise ValueError(f"Sample {sample} has no valid reads in {config['sample_table']}")

        sample_dict[sample] = reads

sample_list = list(sample_dict.keys())
print(sample_list)

# 所使用的组装工具（我们使用megahit进行组装）
assemblers = config['assemblers']  
# ensure at least one valid option
if not ('megahit' in assemblers):  # 确保组装工具为megahit
    sys.exit('Must have at least one valid assembler in config!')

# 辅助函数：将样本对应的文件列表变成megahit需要的命令行参数
def get_megahit_reads_command(reads):
        if len(reads) == 3: # 双端+orphan
            cmd = "-1 {0} -2 {1} -r {2}".format(reads[0], reads[1], reads[2])
        elif len(reads) == 2: # 双端
            cmd = "-1 {0} -2 {1}".format(reads[0], reads[1])
        elif len(reads) == 1: # 单端
            cmd = "--12 {0}".format(reads[0])
        return(cmd)

# 流程： 
# 1、使用megahit进行组装
# 2、把每个样本最终组装的 contigs 列表写成下游工具可直接读取的清单文件

################################################################################
rule all:
    input:
        # megahit组装后的文件
        expand(join(PROJECT_DIR, "00_megahit/{sample}/{sample}.contigs.fa"), sample=sample_list),
        # quast评估组装成果
        expand(join(PROJECT_DIR, "00_megahit/{sample}/quast/report.tsv"), sample=sample_list),
        # quast评估结果汇总
        join(PROJECT_DIR, "00_megahit/quast_report_merged.tsv"),
        # 后续所需要的样本清单
        join(PROJECT_DIR, "contigs_list.txt"),
        # 确保清除中间临时大文件
        join(PROJECT_DIR, "cleaned")

################################################################################
rule megahit:
    input: lambda wildcards: sample_dict[wildcards.sample]
    output: join(PROJECT_DIR, "00_megahit/{sample}/{sample}.contigs.fa")
    threads: 16
    params:
        outdir = join(PROJECT_DIR, "00_megahit/{sample}/"),
        reads_command = lambda wildcards: get_megahit_reads_command(sample_dict[wildcards.sample])
    conda: "/data/home/chenliang/apps/miniconda3/envs/megahit_env"
    benchmark: join(PROJECT_DIR, "00_megahit/{sample}/{sample}_time.txt")
    shell: """
        rm -rf {params.outdir}  # megahit会自动创建文件夹，运行前是不允许该文件夹存在
        megahit \
            --presets meta-sensitive \
            {params.reads_command} \
            -o {params.outdir} \
            -t {threads} \
            --out-prefix {wildcards.sample}
    """

################################################################################
rule quast_megahit:
    input:
        rules.megahit.output
    output:
        join(PROJECT_DIR,"00_megahit/{sample}/quast/report.tsv")
    conda: "/data/home/chenliang/apps/miniconda3/envs/megahit_env"
    params:
        outdir = join(PROJECT_DIR,"00_megahit/{sample}/quast")
    shell: """
        quast -o {params.outdir} {input} --fast
    """

################################################################################
rule combine_megahit_quast_reports:
    input:
        expand(join(PROJECT_DIR, "00_megahit/{sample}/quast/report.tsv"), sample=sample_list),
    output:
        join(PROJECT_DIR, "00_megahit/quast_report_merged.tsv")
    conda: "/data/home/chenliang/apps/miniconda3"
    params:
        sample_names = sample_list,
        assembly_dir = join(PROJECT_DIR, "00_megahit/")
    script: "/data/home/chenliang/project/PRJCA022804/assembly/scripts/combine_quast_reports.py"

################################################################################
# 生成后面functional_profiling所需的样本清单
rule contigs_list_file:
    input: expand(join(PROJECT_DIR, "00_megahit/{sample}/{sample}.contigs.fa"),sample=sample_list)
    output: join(PROJECT_DIR, "contigs_list.txt")
    run:
        outfile = str(output)
        if (os.path.exists(outfile)):
            os.remove(outfile)
        with open(outfile, 'w') as outf:
            outf.writelines(['# Sample\tcontigs\n'])
            for sample in sample_list:
                outline = [sample, '\t'.join([
                join(PROJECT_DIR, "00_megahit/" + sample + "/" + sample + ".contigs.fa")])]
                outf.writelines('\t'.join(outline) + '\n')

################################################################################
# 在其他所有操作完成后运行的清理操作 cleanup to be run after everything else is finished
rule cleanup:
    input: 
        join(PROJECT_DIR, "00_megahit/quast_report_merged.tsv"),
        join(PROJECT_DIR, "contigs_list.txt")
    output: join(PROJECT_DIR, "cleaned")
    params:
        rmdir_1 = join(PROJECT_DIR, '00_megahit'),
    shell: """
        # remove megahit files
        rm -rf {params.rmdir_1}/*/*intermediate_contigs*
        touch {output}
    """