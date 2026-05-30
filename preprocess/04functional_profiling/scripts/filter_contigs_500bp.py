from Bio import SeqIO
import sys
 
long_sequences = [] # Setup an empty list
sample = sys.argv[1] # 样本名
infasta = sys.argv[2] # 输入fasta文件路径
threshold = sys.argv[3] # 长度阈值（bp）
fasta_threshold = sys.argv[4] # 输出文件路径
handle = open(infasta, "r")

for record in SeqIO.parse(handle, "fasta") :
    if len(record.seq) >= int(threshold) :
        # Add this record to our list
        record.id = sample + '+' + record.id
        long_sequences.append(record)
handle.close()
 
#print "Found %i long sequences" % len(long_sequences)
 
output_handle = open(fasta_threshold, "w")
SeqIO.write(long_sequences, output_handle, "fasta")
output_handle.close()
