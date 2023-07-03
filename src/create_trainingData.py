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
        self.parser.add_argument('chromosomes', nargs='+', help='Provide a list of chromosomes to create training data over (include brackets).')
        self.args = self.parser.parse_args()

def createPol3Data(chromosomes, dt="training", step=200, nuc_context=1000):
    pol3_bed_cols_names = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
    pol3_df = pd.read_csv("../data/polr3d.bed", sep="\s+", header=None, names=pol3_bed_cols_names)
    rmsk_df = pd.read_csv("../data/mm10_rmsk.bed", sep="\s+", header=None, names=pol3_bed_cols_names)
    nucOE_df = pd.DataFrame({"Nucleotide": ["a", "c", "g", "t", "n"], "OE": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]})
    j = 1
    
    print("Reading in ChIP files...")
    chip_dfs = []
    chip_names = []
    chip_bed_cols_names = ["Chromosome", "Start", "End", "Name", "Score", "Strand", "signalValue", "pValue", "qValue", "peak"]
    for chip in glob.glob("../data/chip_data/*"):
        tmp_chip_df = pd.read_csv(chip, sep="\s+", header=None, names=chip_bed_cols_names)
        chip_dfs.append(tmp_chip_df)
        chip_names.append(chip.split("/")[-1].split(".")[0])
    del tmp_chip_df
        
    for chrom in chromosomes:
        print(chrom+":")
        print("     Creating necessary directories...")
        output_dir1 = "../data/mlData/"
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
            
        #Process chromosome oe df to create training data
        print("     Processing FASTA sequence...")
        fasta_sequences = SeqIO.parse(open("../chroms/{}.fa".format(chrom)),'fasta')
        for seq in fasta_sequences:
            name, sequence = seq.id, str(seq.seq).lower()
        del fasta_sequences
        chr_df = pd.DataFrame({"Nucleotide": list(sequence)})
        chr_df["Label"] = 0
        pol3_chr_df = pol3_df[pol3_df["Chromosome"] == "{}".format(chrom)]
        rmsk_chr_df = rmsk_df[rmsk_df["Chromosome"] == "{}".format(chrom)]
        
        for row in range(len(rmsk_chr_df)):
            beg_range = rmsk_chr_df.iloc[row]["Start"]
            end_range = rmsk_chr_df.iloc[row]["End"]
            chr_df.loc[beg_range:end_range, "Label"] = 1
        for row in range(len(pol3_chr_df)):
            beg_range = pol3_chr_df.iloc[row]["Start"]
            end_range = pol3_chr_df.iloc[row]["End"]
            chr_df.loc[beg_range:end_range, "Label"] = 2
        del pol3_chr_df
        del rmsk_chr_df
                        
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

        print("     Creating data for model. This may take a while...")
        #Start creating training data
        labels = []
        #Get first and last non-N index
        a_idx = sequence.index("a")
        c_idx = sequence.index("c")
        g_idx = sequence.index("g")
        t_idx = sequence.index("t")
        chr_start_idx = min(a_idx,c_idx,g_idx,t_idx)
        a_idx = sequence.rfind("a")
        c_idx = sequence.rfind("c")
        g_idx = sequence.rfind("g")
        t_idx = sequence.rfind("t")
        chr_end_idx = max(a_idx,c_idx,g_idx,t_idx)
        for i in range(chr_start_idx, chr_end_idx+1, step):
            if i <= chr_end_idx:
                beg_seq = []
                end_seq = []
                
                beg_chip = []
                end_chip = []
                
                start_idx = i - nuc_context
                if start_idx < 0:
                    start_idx = 0
                    n_count = (i - nuc_context) * -1
                    beg_seq = [[0,0,0,0]] * n_count
                    beg_chip = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] * n_count
                end_idx = i+step+nuc_context
                if end_idx > len(chr_df):
                    end_idx = len(chr_df)
                    n_count = (i+step+nuc_context) - len(chr_df)
                    end_seq = [[0,0,0,0]] * n_count
                    end_chip = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]] * n_count
                    
                #Determine labels    
                tmp_df = chr_df[i:i+step]
                grouped_df = tmp_df.groupby("Label").count().reset_index()
                grouped_df.index = grouped_df["Label"]
                try:
                    if grouped_df[grouped_df["Label"] == 2]["Nucleotide"][2] >= 65:
                        label = [1]
                    elif grouped_df[grouped_df["Label"] == 1]["Nucleotide"][1] >= 65:
                        label = [0]
                    else:
                        label = [2]
                except KeyError:
                    try:
                        if grouped_df[grouped_df["Label"] == 1]["Nucleotide"][1] >= 65:
                            label = [0]
                        else:
                            label = [2]
                    except KeyError:
                        label = [2]
                del tmp_df
                del grouped_df
                        
                if label == [0] or label == [1]:
                    #Used to randomly select which nucleotide sequences with label [0] are in datasets
                    if label == [0]:
                        random_num = random.randint(-1000,1)
                    if (label == [0] and random_num > 0) or (label == [1]):                             
                        curr_chr_df = chr_df[start_idx:end_idx].merge(nucOE_df, on=["Nucleotide"], how="left")

                        if beg_seq == [] and end_seq == []:
                            curr_seq = curr_chr_df["OE"].tolist()
                        elif beg_seq == [] and len(end_seq) != 0:
                            curr_seq = curr_chr_df["OE"].tolist() + end_seq
                        elif len(beg_seq) != 0 and end_seq == []:
                            curr_seq = beg_seq + curr_chr_df["OE"].tolist()
                        del curr_chr_df

                        if beg_chip == [] and end_chip == []:
                            curr_chip = chip_df[start_idx:end_idx].values.tolist()
                        elif beg_chip == [] and len(end_chip) != 0:
                            curr_chip = chip_df[start_idx:end_idx].values.tolist() + end_chip
                        elif len(beg_chip) != 0 and end_chip == []:
                            curr_chip = beg_chip + chip_df[start_idx:end_idx].values.tolist()

                        #Save current sequences
                        curr_seq = np.array([curr_seq], dtype=np.uint16)
                        curr_chip = np.array([curr_chip], dtype=np.float64)
                        labels.append(label)
                        if j == 1: 
                            seq_data = curr_seq
                            chip_data = curr_chip
                            j+=1
                        else:
                            seq_data = np.append(seq_data, curr_seq, axis=0)
                            chip_data = np.append(chip_data, curr_chip, axis=0)
                        del curr_seq
                        del curr_chip

        print("Completed {}!".format(chrom))

    print("Finalizing data...")
    labels = np.array(labels)
    true_idx = np.where(labels == [1])[0]
    false_idx = np.where(labels == [0])[0]

    #Shuffle and subset indeces for training and testing datasets

    np.random.shuffle(false_idx)
    false_idx = false_idx[:len(true_idx)]
    seq_data = np.append(seq_data[true_idx], seq_data[false_idx], axis=0)
    chip_data = np.append(chip_data[true_idx], chip_data[false_idx], axis=0)
    
    labels = np.append(labels[true_idx], labels[false_idx], axis=0)
    
    final_idx = np.random.permutation(len(labels))
    seq_data = seq_data[final_idx]
    chip_data = chip_data[final_idx]
    labels = labels[final_idx]
    
    np.savez_compressed("../data/mlData/{}_seqData.npz".format(dt), dna=seq_data, label=labels)
    np.savez_compressed("../data/mlData/{}_chipData.npz".format(dt), chip=chip_data, label=labels)
        
    print("Finished creating data!")
        

def main():

    arguments = Command_line_args()

    createPol3Data(arguments.args.chromosomes, dt="training", step=200, nuc_context=1000)

if __name__ == "__main__":
	main()
