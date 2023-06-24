import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
import glob

import os
import shutil

class Command_line_args(object):
    """
    This class contains all the arguments the user inputs for the class to run.
    Input(s):
    No other inputs needed.
    Output(s):
    None.
    """

    def __init__(self):

        #Command line arguments
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('chromosomes', help='Provide a list of chromosomes to create training data over (include brackets).')
        self.args = self.parser.parse_args()

def createSeqData(chromosomes, step=200, nuc_context=1000):
    pol3_bed_cols_names = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    pol3_df = pd.read_csv("../data/polr3d.bed", sep="\s+", header=None, names=pol3_bed_cols_names)
    for chrom in chromosomes:
        print(chrom+":")
        print("     Creating necessary directories...")
        output_dir1 = "../data/tmp_seqData/"
        output_dir2 = "../data/chr_seqData/"
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
            
        #Process chromosome oe df to create training data
        print("     Processing one-hot encoded dataframe...")
        chr_df = pd.read_csv("../chroms/oe_chroms/{}.csv".format(chrom))
        chr_df["Label"] = 0
        pol3_chr_df = pol3_df[pol3_df["Chromosome"] == "{}".format(chrom)]
        for row in range(len(pol3_chr_df)):
            beg_range = pol3_chr_df.iloc[row]["Start"]
            end_range = pol3_chr_df.iloc[row]["End"]
            chr_df.loc[beg_range:end_range, "Label"] = 1

        print("     Creating training data. This may take a while...")
        #Start creating training data
        labels = []
        file_names = []
        final_data = []
        j = 1
        #Get first and last non-N index
        fasta_sequences = SeqIO.parse(open("../chroms/{}.fa".format(chrom)),'fasta')
        for seq in fasta_sequences:
            name, sequence = seq.id, str(seq.seq)
        a_idx = sequence.lower().index("a")
        c_idx = sequence.lower().index("c")
        g_idx = sequence.lower().index("g")
        t_idx = sequence.lower().index("t")
        chr_start_idx = min(a_idx,c_idx,g_idx,t_idx)
        a_idx = sequence.lower().rfind("a")
        c_idx = sequence.lower().rfind("c")
        g_idx = sequence.lower().rfind("g")
        t_idx = sequence.lower().rfind("t")
        chr_end_idx = max(a_idx,c_idx,g_idx,t_idx)
        for i in range(chr_start_idx, chr_end_idx+1, step):
            if i <= chr_end_idx:
                beg_seq = []
                end_seq = []
                
                start_idx = i - nuc_context
                if start_idx < 0:
                    start_idx = 0
                    n_count = (i - nuc_context) * -1
                    beg_seq = [[0,0,0,0]] * n_count
                end_idx = i+step+nuc_context
                if end_idx > len(chr_df):
                    end_idx = len(chr_df)
                    n_count = (i+step+nuc_context) - len(chr_df)
                    end_seq = [[0,0,0,0]] * n_count

                if beg_seq == [] and end_seq == []:
                    training_seq = chr_df[start_idx:end_idx].drop(columns=["Unnamed: 0", "Label"]).to_numpy()
                elif beg_seq == [] and len(end_seq) != 0:
                    training_seq = chr_df[start_idx:end_idx].drop(columns=["Unnamed: 0", "Label"]).to_numpy() + np.array(end_seq)
                elif len(beg_seq) != 0 and end_seq == []:
                    training_seq = beg_seq + chr_df[start_idx:end_idx].drop(columns=["Unnamed: 0", "Label"]).values.tolist()
                #Determine labels    
                tmp_df = chr_df[start_idx:end_idx]
                grouped_df = tmp_df.groupby("Label").count().reset_index()
                try:
                    if grouped_df[grouped_df["Label"] == 1]["Unnamed: 0"][1] >= 65:
                        labels.append([1])
                    else:
                        labels.append([0])
                except KeyError:
                    labels.append([0])
                #Save temp files for later concatenation
                training_seq = np.array([training_seq], dtype=np.uint16)
                if j == 1: 
                    training_data = training_seq
                else:
                    training_data = np.append(training_data, training_seq, axis=0)
                if j % 50 == 0:
                    np.savez_compressed("../data/tmp_seqData/tmp_{}.npz".format(i), training_data)
                    file_names.append("../data/tmp_seqData/tmp_{}.npz".format(i))
                    j = 1
                else:
                    j+=1
        del training_data
        
        print("     Finalizing training data...")

        #Open numpy npz files and memmap them to reduce memory usage
        fpath = "../data/chr_seqData/{}_seqData.dat".format(chrom)
        rows = 0
        cols = None
        dtype = None
        for data_file in file_names:
            with np.load(data_file) as data:
                for item in data.files:
                    chunk = data[item]
                    rows += chunk.shape[0]
                    cols = chunk.shape[1]
                    elements = chunk.shape[2]
                    dtype = chunk.dtype
                
        merged = np.memmap(fpath, dtype=dtype, mode='w+', shape=(rows, cols, elements))
        idx = 0
        for data_file in file_names:
            with np.load(data_file) as data:
                for item in data.files:
                    chunk = data[item]
                    merged[idx:idx + len(chunk)] = chunk
                    idx += len(chunk)
                
        #Save chr data
        labels = np.array(labels)
        fpath2 = "../data/chr_seqData/{}_labelsData.npz".format(chrom)
        np.savez_compressed(fpath2, labels=labels)
        
        #Delete temp directories
        dir = '../data/tmp_seqData/'
        shutil.rmtree(dir)
        
        print("Completed {}!".format(chrom))
        print("Memmap object dimensions:", rows, columns, elements)

    print("Finished creating training data by chromosome!")

def createChipData(chromosomes, step=200, nuc_context=1000):
    print("Reading in chip files...")
    chip_dfs = []
    chip_names = []
    chip_bed_cols_names = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "signalValue", "pValue", "qValue", "peak"]
    for chip in glob.glob("../data/chip_data/*"):
        tmp_chip_df = pd.read_csv(chip, sep="\s+", header=None, names=chip_bed_cols_names)
        chip_dfs.append(tmp_chip_df)
        chip_names.append(chip.split("/")[-1].split(".")[0])
    
    for chrom in chromosomes:
        print(chrom+":")
        print("     Creating necessary directories...")
        output_dir1 = "../data/tmp_chipData/"
        output_dir2 = "../data/chr_chipData/"
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
            
        #Process chromosome oe df to create training data
        print("     Reading in one-hot encoded dataframe...")
        chr_df = pd.read_csv("../chroms/oe_chroms/{}.csv".format(chrom))
            
        #Create ChIP df that is ready to be converted to numpy training data
        print("     Processing ChIP data...")
        chip_df = pd.DataFrame(index=np.arange(len(chr_df))).reset_index().drop(columns="index")
        chip_idx = 0
        for chip in chip_dfs:
            tmp_chip_chr_df = chip[chip["Chromosome"] == "{}".format(chrom)]
            chip_df[chip_names[chip_idx]] = 0.
            
            for row in range(len(tmp_chip_chr_df)):
                beg_range = tmp_chip_chr_df.iloc[row]["Start"]
                end_range = tmp_chip_chr_df.iloc[row]["End"]
                chip_df.loc[beg_range:end_range, chip_names[chip_idx]] = tmp_chip_chr_df.iloc[row]["signalValue"]
            chip_idx+=1
                        

        print("     Creating training data. This may take a while...")
        #Start creating training data
        file_names = []
        final_data = []
        j = 1
        #Get first and last non-N index
        fasta_sequences = SeqIO.parse(open("../chroms/{}.fa".format(chrom)),'fasta')
        for seq in fasta_sequences:
            name, sequence = seq.id, str(seq.seq)
        a_idx = sequence.lower().index("a")
        c_idx = sequence.lower().index("c")
        g_idx = sequence.lower().index("g")
        t_idx = sequence.lower().index("t")
        chr_start_idx = min(a_idx,c_idx,g_idx,t_idx)
        a_idx = sequence.lower().rfind("a")
        c_idx = sequence.lower().rfind("c")
        g_idx = sequence.lower().rfind("g")
        t_idx = sequence.lower().rfind("t")
        chr_end_idx = max(a_idx,c_idx,g_idx,t_idx)
        for i in range(chr_start_idx, chr_end_idx+1, step):
            if i <= chr_end_idx:
                beg_seq = []
                end_seq = []
                
                start_idx = i - nuc_context
                if start_idx < 0:
                    start_idx = 0
                    n_count = (i - nuc_context) * -1
                    beg_seq = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] * n_count
                end_idx = i+step+nuc_context
                if end_idx > len(chr_df):
                    end_idx = len(chr_df)
                    n_count = (i+step+nuc_context) - len(chr_df)
                    end_seq = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] * n_count

                if beg_seq == [] and end_seq == []:
                    training_seq = chip_df[start_idx:end_idx].to_numpy()
                elif beg_seq == [] and len(end_seq) != 0:
                    training_seq = chip_df[start_idx:end_idx].to_numpy() + np.array(end_seq)
                elif len(beg_seq) != 0 and end_seq == []:
                    training_seq = beg_seq + chip_df[start_idx:end_idx].values.tolist()

                #Save temp files for later concatenation
                training_seq = np.array([training_seq], dtype=np.float32)
                if j == 1: 
                    training_data = training_seq
                else:
                    training_data = np.append(training_data, training_seq, axis=0)
                if j % 50 == 0:
                    
                    np.savez_compressed("../data/tmp_chipData/tmp_{}.npz".format(i), training_data)
                    file_names.append("../data/tmp_chipData/tmp_{}.npz".format(i))
                    j = 1
                else:
                    j+=1
        del training_data
        
        print("     Finalizing training data...")
        #Open numpy npz files and memmap them to reduce memory usage
        fpath = "../data/chr_chipData/{}_chipData.dat".format(chrom)
        rows = 0
        cols = None
        dtype = None
        for data_file in file_names:
            with np.load(data_file) as data:
                for item in data.files:
                    chunk = data[item]
                    rows += chunk.shape[0]
                    cols = chunk.shape[1]
                    elements = chunk.shape[2]
                    dtype = chunk.dtype
                
        merged = np.memmap(fpath, dtype=dtype, mode='w+', shape=(rows, cols, elements))
        idx = 0
        for data_file in file_names:
            with np.load(data_file) as data:
                for item in data.files:
                    chunk = data[item]
                    merged[idx:idx + len(chunk)] = chunk
                    idx += len(chunk)
        
        #Delete temp directories
        dir = '../data/tmp_chipData/'
        shutil.rmtree(dir)
        
        print("Completed {}!".format(chrom))
        print("Memmap object dimensions:", rows, columns, elements)

    print("Finished creating training data by chromosome!")

def main():

	# all_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14",
	#                   "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"]

    arguments = Command_line_args()

	createSeqData(arguments.args.chromosomes, step=200, nuc_context=1000)
	createChipData(arguments.args.chromosomes, step=200, nuc_context=1000)

if __name__ == "__main__":
	main()
