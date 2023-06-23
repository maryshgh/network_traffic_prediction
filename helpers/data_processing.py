__author__ = 'Maryam Shabani'
import numpy as np

# Creating spatial data from raw dataset
def build_spatial_data(data, data_length, r, start_row, end_row, start_col, end_col, area_length):
  # initialize the spatial data 
  data_spatial = np.zeros((data_length,r,r))
  for t in range(data_length):
      # extract the data of the all cells in the area of the interest at time t
      data_temp = data[data[:,1]==t]
      for i in range(start_row,end_row):
          for j in range(start_col,end_col):
              data_spatial[t,i-start_row,j-start_col] = data_temp[data_temp[:,0]==area_length*i+j+1, 2]

  min_data=np.min(data_spatial)
  max_data = np.max(data_spatial)
  return data_spatial, min_data, max_data

# creating 3D data
def build_sequence_data(data, recurrent_length, r, area_borders, data_length, rec_out, area_length):
  data_spatial, min_data, max_data = build_spatial_data(data, data_length, r, area_borders[0], area_borders[1], area_borders[2], area_borders[3], area_length)
              
  for t in range(recurrent_length,data_length-rec_out):
    if t==recurrent_length:
      data_t_ij=data_spatial[t-recurrent_length:t,:,:].reshape(1,recurrent_length,r,r,1)
      data_y = data_spatial[t:t+rec_out,:,:].reshape(1,rec_out,r,r,1)
    else:
      data_t_ij = np.concatenate((data_t_ij,data_spatial[t-recurrent_length:t,:,:].reshape(1,recurrent_length,r,r,1)),axis=0)
      data_y = np.concatenate((data_y,data_spatial[t:t+rec_out,:,:].reshape(1,rec_out,r,r,1)),axis=0)
  
  return data_t_ij, data_y, min_data, max_data
