
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import random
import os
from time import time
import traceback
import sys
sys.path.append('/rds/project/rds-a1NGKrlJtrw/LLPS/') 
from alphafold.common import residue_constants


def read_picklefile(filename):
  return pickle.load(open(filename,'rb'))

class DataGenerator():
  def __init__(self,data_paths,data_config,crop_lower,crop_upper):
    print("Starting loading raw data")
    self.data={}
    self.crop_size=crop_upper
    crop_size=crop_upper
    pair_channels=data_config.pair_channels
    single_channels=data_config.single_channels
    protein_seqs={}
    protein_dirs={}
    protein_feats={}
    for data_name,data_path in data_paths.items():
      data=pd.read_csv(data_path)
      protein_set=set(data['id_1']) | set(data['id_2'])
      time_begin=time()
      seq_len_dict={}
      for i,row in data.iterrows():
        seq_len_dict[row['id_1']]=int(row['len_1'])
        seq_len_dict[row['id_2']]=int(row['len_2'])
        protein_seqs[row['id_1']]=row['seq_1']
        protein_seqs[row['id_2']]=row['seq_2']
        protein_dirs[row['id_1']]=row['dir_1']
        protein_dirs[row['id_2']]=row['dir_2']
      print(f"Before processing data {data_name}:  Protein: {len(protein_set)}")
      for seq_id in protein_set:
        if seq_id in protein_feats.keys():
          continue
        try:
          # with open(protein_fasta_path+seq_id+'.fasta') as f:
          #   seq=(f.readlines()[1]).strip()
          resi_num=seq_len_dict[seq_id]
          seq=protein_seqs[seq_id]
          protein_dir=protein_dirs[seq_id]
          if resi_num > crop_upper or resi_num<= crop_lower:continue
          pair_repr_files = [f for f in os.listdir(protein_dir) if f.startswith(seq_id + '_pair_repr') and f.endswith('.npy')]
          single_repr_files = [f for f in os.listdir(protein_dir) if f.startswith(seq_id + '_single_repr') and f.endswith('.npy')]
          structure_repr_files = [f for f in os.listdir(protein_dir) if f.startswith(seq_id + '_structure_repr') and f.endswith('.npy')]
          if not pair_repr_files or not single_repr_files or not structure_repr_files:
            print(f"Warning: No representation files found for {seq_id} in {protein_dir}")
            continue
          if len(pair_repr_files) > 1 or len(single_repr_files) > 1 or len(structure_repr_files) > 1:
            print(f"Warning: Multiple representation files found for {seq_id} in {protein_dir}, using the first one.")
          pair=np.load(protein_dir+'/'+pair_repr_files[0])
          single=np.load(protein_dir+'/'+single_repr_files[0])
          struc=np.load(protein_dir+'/'+structure_repr_files[0])
          sequence=residue_constants.sequence_to_onehot(
          sequence=seq,
          mapping=residue_constants.restype_order_with_x,
          map_unknown_to_x=True)

          single=single[:resi_num,:]
          struc=struc[:resi_num,:]
          pair=pair[:resi_num,:resi_num,:]
          protein_feat={'pair':pair,'single':single,'struc':struc,'sequence':sequence,'resi_num':resi_num}
          protein_feats[seq_id]=protein_feat
        except Exception as e:
          print('Error: cannot open sequence',seq_id)
          print(e)
          traceback.print_exc()
      data=data[data['id_1'].isin(protein_feats.keys())]
      data=data[data['id_2'].isin(protein_feats.keys())]
      self.data[data_name]=data
      print(f"After processing protein {data_name}:  Protein: {len(protein_feats)} Time usage:{time()-time_begin}")
      time_begin=time()

    self.protein_feats=protein_feats

    self.output_types= {'pair_act':np.float32,'msa_act':np.float32,'struc_act':np.float32, 'sequence':np.int8, 'msa_mask':np.int8,'pair_mask':np.int8,
      'resi_num':np.int16, 'pair_act_2':np.float32,'msa_act_2':np.float32,'struc_act_2':np.float32, 'sequence_2':np.int8,'msa_mask_2':np.int8,'pair_mask_2':np.int8,
      'resi_num_2':np.int16,'solubility':np.float32,'id':np.int64}

    self.output_shapes={'pair_act':[crop_size,crop_size,pair_channels],'msa_act':[crop_size,single_channels],'struc_act':[crop_size,single_channels],
      'msa_mask':[crop_size],'pair_mask':[crop_size,crop_size],'resi_num':[],'sequence':[crop_size,21],
      'pair_act_2':[crop_size,crop_size,pair_channels],'msa_act_2':[crop_size,single_channels],'struc_act_2':[crop_size,single_channels],
      'msa_mask_2':[crop_size],'pair_mask_2':[crop_size,crop_size],'resi_num_2':[],'sequence_2':[crop_size,21],
      'solubility':[],'id':[]}
    self.data_config=data_config

  def generate(self,data_name,shuffle=False):
    data_config=self.data_config
    crop_size=self.crop_size
    batch_size=data_config.training.batch_size
    data=self.data[data_name]
    if shuffle:
      t=time()
      print(f'Random seed for shuffle:{t}')
      data=data.sample(frac = 1,random_state=int(t))
    data_len=int(len(data)//batch_size)*batch_size
    data=data.iloc[:data_len]
    print(f"Data length: {len(data)}")
    for i,row in data.iterrows():
      seq_id=row['id_1']
      seq_id_2=row['id_2']
      id=row['raw_id']
      if seq_id not in self.protein_feats.keys() or seq_id_2 not in self.protein_feats.keys():
        print(f"Warning: {seq_id} or {seq_id_2} not in protein feats, skip")
        continue
      protein_feat=self.protein_feats[seq_id]
      protein_feat_2=self.protein_feats[seq_id_2]
      single=protein_feat['single']
      struc=protein_feat['struc']
      pair=protein_feat['pair']
      sequence=protein_feat['sequence']
      resi_num=protein_feat['resi_num']
      length=single.shape[0]

      single_2=protein_feat_2['single']
      struc_2=protein_feat_2['struc']
      pair_2=protein_feat_2['pair']
      sequence_2=protein_feat_2['sequence']
      resi_num_2=protein_feat_2['resi_num']
      length_2=single_2.shape[0]

      solubility =0.0

      if resi_num > crop_size or resi_num_2>crop_size:
        print(f"Warning: {seq_id} or {seq_id_2} length {resi_num} or {resi_num_2} is larger than crop size {crop_size}, skip")
        continue
      if length > crop_size:
        pair=pair[:crop_size,:crop_size,:]
        single=single[:crop_size,:]
        struc=struc[:crop_size,:]
        sequence=sequence[:crop_size,:]
      length=single.shape[0]
      if length_2 > crop_size:
        pair_2=pair_2[:crop_size,:crop_size,:]
        single_2=single_2[:crop_size,:]
        struc_2=struc_2[:crop_size,:]
        sequence_2=sequence_2[:crop_size,:]
      length_2=single_2.shape[0]
      msa_mask=np.ones(resi_num)
      pair_mask=np.ones((resi_num,resi_num))
      msa_mask_2=np.ones(resi_num_2)
      pair_mask_2=np.ones((resi_num_2,resi_num_2))

      pair_mask=np.pad(pair_mask,((0,crop_size-resi_num),(0,crop_size-resi_num)),'constant',constant_values=(0,0))
      msa_mask=np.pad(msa_mask,(0,crop_size-resi_num),'constant',constant_values=(0,0))
      single=np.pad(single,((0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
      struc=np.pad(struc,((0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
      sequence=np.pad(sequence,((0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
      pair=np.pad(pair,((0,crop_size-length),(0,crop_size-length),(0,0)),'constant',constant_values=(0.0,0.0))
      
      pair_mask_2=np.pad(pair_mask_2,((0,crop_size-resi_num_2),(0,crop_size-resi_num_2)),'constant',constant_values=(0,0))
      msa_mask_2=np.pad(msa_mask_2,(0,crop_size-resi_num_2),'constant',constant_values=(0,0))
      single_2=np.pad(single_2,((0,crop_size-length_2),(0,0)),'constant',constant_values=(0.0,0.0))
      struc_2=np.pad(struc_2,((0,crop_size-length_2),(0,0)),'constant',constant_values=(0.0,0.0))
      sequence_2=np.pad(sequence_2,((0,crop_size-length_2),(0,0)),'constant',constant_values=(0.0,0.0))
      pair_2=np.pad(pair_2,((0,crop_size-length_2),(0,crop_size-length_2),(0,0)),'constant',constant_values=(0.0,0.0))

      pair_mask= np.asanyarray(pair_mask,dtype=np.int8)
      msa_mask= np.asanyarray(msa_mask,dtype=np.int8)    
      single= np.asanyarray(single,dtype=np.float32)
      struc= np.asanyarray(struc,dtype=np.float32)
      pair= np.asanyarray(pair,dtype=np.float32)   
      sequence=np.asanyarray(sequence,dtype=np.int8)  

      pair_mask_2= np.asanyarray(pair_mask_2,dtype=np.int8)
      msa_mask_2= np.asanyarray(msa_mask_2,dtype=np.int8)    
      single_2= np.asanyarray(single_2,dtype=np.float32)
      struc_2= np.asanyarray(struc_2,dtype=np.float32)
      pair_2= np.asanyarray(pair_2,dtype=np.float32)   
      sequence_2=np.asanyarray(sequence_2,dtype=np.int8)  

      solubility= np.asanyarray(solubility,dtype=np.float32)
      id=int(id)
      id=np.asanyarray(id,dtype=np.int64)
      #geneid=np.asanyarray(geneid,dtype=np.float32)
      assert single.shape==(crop_size,data_config.single_channels)
      assert sequence.shape==(crop_size,21)
      assert sequence_2.shape==(crop_size,21)
      assert pair.shape==(crop_size,crop_size,data_config.pair_channels)
      assert msa_mask.shape==(crop_size,)
      assert pair_mask.shape==(crop_size,crop_size)
      feat={'pair_act':pair,'msa_act':single,'struc_act':struc,'msa_mask':msa_mask,'pair_mask':pair_mask,'sequence':sequence,'resi_num':resi_num,
            'pair_act_2':pair_2,'msa_act_2':single_2,'struc_act_2':struc_2,'msa_mask_2':msa_mask_2,'pair_mask_2':pair_mask_2,'sequence_2':sequence_2,'resi_num_2':resi_num_2,
            'id':id,'solubility':solubility}
      yield feat

