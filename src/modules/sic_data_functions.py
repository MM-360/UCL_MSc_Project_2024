import dill
import numpy as np
import torch
import pandas as pd
import xlsxwriter


def get_ice_data(data_loc = r'C:/Project/data/data314_years_1989_2023.pkl' , return_mask = True, return_all = False):
    with open(data_loc, 'rb') as f:
        m, Y, Y_mean_month, Y_mean_week, x2, y2 = dill.load(f)
    # mask = array of T/F plotting out the land mass of Antartica
    #DATA = Y
    #x = x2
    #y = y2
    #del Y_mean_month, Y_mean_week

    if return_all:
        return Y,  ~m, x2,y2,Y_mean_month,Y_mean_week
    else:
        if return_mask:
            return Y,  ~m  
        else:
            return Y


def thin_data(data_input, mask_input, step_thin, print_shape = True):
    # step parameter
    #step_thin = step_size

    # ~ is a binary operator flips 1 to 0 and vice versa
    mask = mask_input[::step_thin, ::step_thin]
    # reducing the data set size with the step paramter only for model searching/training

    DATA = []
    for year in range(len(data_input)):
        #y0 = Y[year][:, ::step_thin, ::step_thin]
        DATA.append(data_input[year][:, ::step_thin, ::step_thin])

    #check shape
    if print_shape:
        DATA[0].shape
    
    return DATA, mask


def iiee_calc(c_m, c_o, c_e = 0.15, iiee_only =True):

    """
    Torch implementation of Integrated Ice Edge error calcuation

    c_m = model image
    c_o = obersevation image

    c_e = edge variable, set to 0.15 as per ....

    Returns:

    """
    model_edge  = c_m >= c_e
    observed_edge = c_o >= c_e

    a_pos = torch.logical_and(model_edge, ~observed_edge)
    a_neg = torch.logical_and(~model_edge, observed_edge)

    a_pos_area = torch.sum(a_pos)
    a_neg_area = torch.sum(a_neg)

    iiee = a_pos_area + a_neg_area
    bias = a_pos_area - a_neg_area

    if iiee_only:
        return iiee
    else:
        return iiee, bias, a_pos_area, a_neg_area
    

batched_iiee = torch.func.vmap(iiee_calc)
batched_min = torch.func.vmap(torch.min)
batched_max = torch.func.vmap(torch.max)


def normalise_image_single(in_data):
    return (in_data - in_data.min())/(in_data.max() - in_data.min())


def normalise_imagev2(in_data):

    if in_data.dim() > 3:
        in_data1 = in_data.permute(1,0,2,3)
        minval = in_data1[0].min(1)[0].min(1)[0]
        maxval = in_data1[0].max(1)[0].max(1)[0]
    else:
        in_data1 = in_data
        minval = in_data1.min(1)[0].min(1)[0]
        maxval = in_data1.max(1)[0].max(1)[0]

    return (in_data - minval[:,None,None,None])/(maxval[:,None,None,None] - minval[:,None,None,None])


def normalise_image(in_data):

    if in_data.dim() > 3:
        in_data1 = in_data.permute(1,0,2,3)
        minval = batched_min(in_data1[0])
        maxval = batched_max(in_data1[0])
    else:
        in_data1 = in_data
        minval = batched_min(in_data1)
        maxval = batched_max(in_data1)

    return (in_data - minval[:,None,None,None])/(maxval[:,None,None,None] - minval[:,None,None,None])


def numpy_mse(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean()


def yoy_stats(in_data):
    reshaped_data = in_data.reshape(int(in_data[0]/365), 365, 314,314)
    reshaped_data_mean = reshaped_data.mean(axis =0)
    reshaped_data_std = reshaped_data.std(axis =0)
    return reshaped_data_mean, reshaped_data_std

def yoy_stats_torch(in_data):
    reshaped_data = in_data.reshape(int(in_data[0]/365), 365, 314,314)
    reshaped_data_mean = reshaped_data.mean(dim =0)
    reshaped_data_std = reshaped_data.std(dim =0)
    return reshaped_data_mean, reshaped_data_std


def test_data_stats(test_data,clamp_yn, norm_yn, encoder, decorder):

    avg_stats = []

    X_test_all = torch.Tensor(test_data)[:,None,...]

    enc_test = encoder(X_test_all)

    if clamp_yn == True:
        dec_test = decorder(enc_test).clamp(min=0, max= 1)
    elif norm_yn == True:
        dec_test = normalise_image(decorder(enc_test))#*mask_tensor[:730,:,:,:])
    else:
        dec_test = decorder(enc_test)
         
    X_test, dec_test = X_test_all.permute(1,0,2,3)[0], dec_test.permute(1,0,2,3)[0]

    mse_per_img = ((dec_test - X_test)**2).sum(dim=(1,2))/(X_test.shape[2]**2)

    mae_per_img = (torch.abs(dec_test - X_test)).sum(dim=(1,2))/(X_test.shape[2]**2)

    avg_mse = mse_per_img.sum()/mse_per_img.shape[0]

    avg_mae = mae_per_img.sum()/mae_per_img.shape[0]  

    avg_iiee = iiee_calc(dec_test, X_test)/X_test.shape[0]
    
    iiee_per_img = batched_iiee(dec_test, X_test)

    print("Total MSE: {:.4f}, RMSE: {:.4f} MAE: {:.4f} Avg iiee: {:.4f}".format(avg_mse.detach().numpy(),np.sqrt(avg_mse.detach().numpy()), avg_mae.detach().numpy(), avg_iiee))

    avg_stats.append([avg_mse.detach().numpy().item(),avg_mae.detach().numpy().item(),avg_iiee.detach().numpy().item()])

    return [avg_stats, mse_per_img.detach().numpy(), mae_per_img.detach().numpy(), iiee_per_img.detach().numpy()]



def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fk' % (x * 1e-4)

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fm' % (x * 1e-6)

def get_climate_avg(input_arr, c_window, normalise = True):
    temp_arr = input_arr[-c_window*365:,...]
    temp_mean = temp_arr.reshape(c_window,365,-1).mean(dim=0)
    temp_arr_no_mean = temp_arr.reshape(c_window,365,-1) - temp_mean
    temp_return = temp_arr_no_mean.reshape(c_window*365,-1)

    if normalise:
        temp_min = temp_return.min(dim = 0)[0]
        temp_max = temp_return.max(dim = 0)[0]
        temp_return_norm = (temp_return - temp_min) /(temp_max- temp_min)
        
        return temp_return, temp_return_norm, temp_mean,  temp_max, temp_min
    else:
        return temp_return, temp_mean


def get_climate_avg_img(input_arr, c_window, normalise = False):
    temp_arr = input_arr[-c_window*365:,...]
    temp_mean = temp_arr.reshape(c_window,365, 314,314).mean(dim=0)
    temp_arr_no_mean = temp_arr.reshape(c_window,365,314,314) - temp_mean
    temp_return = temp_arr_no_mean.reshape(c_window*365,314,314)

    if normalise:
        temp_min = temp_return.min(dim = 0)[0]
        temp_max = temp_return.max(dim = 0)[0]
        temp_return_norm = (temp_return - temp_min) /(temp_max- temp_min)
        
        return temp_return, temp_return_norm, temp_mean,  temp_max, temp_min
    else:
        return temp_return, temp_mean
    
def multiple_dfs_to_excel(df_list, sheet_list, file_name):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)   
    writer.close()
