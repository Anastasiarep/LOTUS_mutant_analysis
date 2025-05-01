# -*- coding: utf-8 -*-
"""
Title: Quantification of Spatial Molecule Enrichment and Colocalisation from Fluorescence Micrographs

Description:
This script processes multichannel fluorescence micrographs of fixed samples 
(i.e. Drosophila oocytes or embryos) to quantify spatial protein and mRNA enrichment 
along the anterior-posterior axis and to assess colocalisation between markers. 

Key features include:
- Extraction and normalisation (z-scoring) of fluorescence intensities per channel
- Generation of spatial intensity profiles (linescans)
- Computation of integrated anterior enrichment
- Correlation analyses (e.g., Pearson's r) between channels in regions of interest
- Statistical comparisons and plotting of results across genotypes or experimental conditions

Author: Anastasia Repouliou
"""

# Import required libraries for data processing, image analysis, and plotting
import numpy as np                                 # numerical operations
import matplotlib.pyplot as plt                    # plotting
import os                                          # file path handling
import skimage                                     # image processing
from skimage import io                             # image I/O
import pandas as pd                                # dataframe operations
import scipy.stats as st                           # statistics
from scipy.stats import pearsonr, norm             # specific statistical functions
import scipy.ndimage as ndimage                    # image filtering
from matplotlib.pyplot import cm                   # colormap handling
from matplotlib.cm import get_cmap                 # alternative colormap access
from scipy.interpolate import interp1d             # interpolation for smooth profiles
from matplotlib.font_manager import FontProperties # font customisation for plots


# Function to calculate spatial intensity profiles and correlation metrics for a given micrograph
# Inputs:
# - experiment: 'linescan' or 'anterior_metrics', defines the analysis type
# - file_prefix: name of the micrograph file without the extension
# - sample_dir: directory containing micrograph files
# - PS: if True, plot spatial intensity profiles (used with 'linescan')
# Returns:
# - A single-row DataFrame containing spatial metrics or enrichment and correlation values
def return_multichannel_spatial_channel_intensity(experiment, file_prefix, sample_dir, PS = False):
    # Load binary mask and multichannel image for the specified micrograph
    Mask = np.array(io.imread(sample_dir + '\\Masks\\mask_'+file_prefix+'.tif'))
    Mask[Mask > 0] = 1
    Img = np.array(io.imread(sample_dir+'\\Rotated\\rotated_'+file_prefix+'.tif'), dtype = np.float64)

    #--------------------------#
    # Optional: display full uncropped mask and all available channels
    # fig,ax = plt.subplots(ncols=Img.shape[2]+1,figsize=(12,3),sharey=True)
    # ax[0].imshow(Mask,cmap='binary')
    # ax[1].imshow(Img[:,:,0])
    # ax[2].imshow(Img[:,:,1])
    # ax[3].imshow(Img[:,:,2])
    # if (Img.shape[2] == 4):
    #     ax[4].imshow(Img[:,:,3])
    # plt.show()
    #--------------------------#

    # Identify bounding box of the mask in X (columns) and Y (rows)
    XPixels = np.sum(Mask,axis=0)
    p0 = np.sum(np.cumsum(XPixels) == 0)
    p1 = np.sum((np.cumsum(XPixels) - np.sum(XPixels)) > 0)
    YPixels = np.sum(Mask,axis=1)
    l0 = np.sum(np.cumsum(YPixels) == 0)
    l1 = np.sum((np.cumsum(YPixels) - np.sum(YPixels)) > 0)

    #--------------------------#
    # Optional: plot the 1D mask projection along X to verify cropping boundaries
    # fig,ax = plt.subplots(figsize=(4,4))
    # ax.plot(range(len(XPixels)),np.sum(Mask,axis=0),'.')
    # ax.axvline(p0,linestyle='--',color='r')
    # ax.axvline(p1,linestyle='--',color='g')
    # plt.show()
    #--------------------------#
    
    # Crop both image and mask to bounding box
    Mask_cropped = Mask[l0:(l1+1),p0:(p1+1)]
    Img_cropped = Img[l0:(l1+1),p0:(p1+1),:]
    XPixels_cropped = XPixels[p0:(p1+1)]
    YPixels_cropped = YPixels[l0:(l1+1)]

    #--------------------------#
    # Optional: display cropped mask and image channels for inspection
    # fig,ax = plt.subplots(ncols=Img.shape[2]+1,figsize=(9,3),sharey=True)
    # ax[0].imshow(Mask_cropped,cmap='binary')
    # ax[1].imshow(Img_cropped[:,:,0])
    # ax[2].imshow(Img_cropped[:,:,1])
    # ax[3].imshow(Img_cropped[:,:,2])
    # if (Img.shape[2] == 4):
    #     ax[4].imshow(Img_cropped[:,:,3])
    #     # ax[4].vlines(x=post_edge, ymin=0, ymax=700, color='r', linewidth=1)
    #     # ax[4].vlines(x=post2_edge, ymin=0, ymax=700, color='r', linewidth=1)
    # plt.show()
    #--------------------------#
    
    # Extract masked intensity values for each channel
    y1,y2,y3 = Img_cropped[:,:,0][Mask_cropped > 0],Img_cropped[:,:,1][Mask_cropped > 0],Img_cropped[:,:,2][Mask_cropped > 0]
    if (Img.shape[2] == 4):
        y4 = Img_cropped[:,:,3][Mask_cropped > 0]

    # Compute per-channel mean and standard deviation from masked region
    mu1,mu2,mu3,sig1,sig2,sig3 = np.mean(y1),np.mean(y2),np.mean(y3),np.std(y1),np.std(y2),np.std(y3)
    if (Img.shape[2] == 4):
        mu4,sig4 = np.mean(y4),np.std(y4)

    # Compute per-pixel z-scores for each channel (within mask)
    z1,z2,z3 = Mask_cropped*((Img_cropped[:,:,0] - mu1)/sig1), Mask_cropped*((Img_cropped[:,:,1] - mu2)/sig2), Mask_cropped*((Img_cropped[:,:,2] - mu3)/sig3)
    if (Img.shape[2] == 4):
        z4 = Mask_cropped*((Img_cropped[:,:,3] - mu4)/sig4)
        
    #--------------------------#
    # Optional: display computed z-score images
    # fig,ax = plt.subplots(ncols=Img.shape[2]+1,figsize=(9,3),sharey=True)
    # ax[0].imshow(Mask_cropped,cmap='binary')
    # ax[1].imshow(z1)
    # ax[2].imshow(z2)
    # ax[3].imshow(z3)
    # if (Img.shape[2] == 4):
    #     ax[4].imshow(z4)
    # plt.show()
    #-------------------------#
    
    # Process for extracting z-score linescans across the anterior-posterior axis
    if experiment == 'linescan':
        # Collapse z-scores along Y to generate anterior-posterior linescans
        A = np.sum(z1,axis=0)/XPixels_cropped
        B = np.sum(z2,axis=0)/XPixels_cropped
        C = np.sum(z3,axis=0)/XPixels_cropped
        if (Img.shape[2] == 4):
            D = np.sum(z4,axis=0)/XPixels_cropped
        
        # Interpolate to fixed resolution along AP axis (300 points)
        x_ = np.arange(0,len(A))/(len(A)-1)
        X = np.linspace(0,1,300)
        fit_func_A,fit_func_B,fit_func_C = interp1d(x_,A),interp1d(x_,B),interp1d(x_,C)
        if (Img.shape[2] == 4):
            fit_func_D = interp1d(x_,D)
        A_,B_,C_ = fit_func_A(X),fit_func_B(X),fit_func_C(X)
        if (Img.shape[2] == 4):
            D_ = fit_func_D(X)
            
        # Optional: plot z-score linescans across AP axis
        if PS == True:
            fig,ax = plt.subplots()
            ax.set_xticks([0,0.5,1.0])
            ax.plot(X,B_,'g.',label='channel_2')
            ax.plot(X,C_,'b.',label='channel_3')
            if (Img.shape[2] == 4):
                ax.plot(X,D_,'r.',label='channel_4')
            ax.legend(loc='best')
            plt.show()
        
        # Concatenate profile data into a single array
        if (Img.shape[2] == 4):
            all_array = np.concatenate((X, A_, B_, C_, D_), axis=None)
        else:
            all_array = np.concatenate((X, A_, B_, C_), axis=None)
        df_row = pd.DataFrame([all_array], index=[file_prefix])
        return(df_row)
    
    # Process for calculating anterior enrichment and pairwise correlation
    if experiment == 'anterior_metrics':

        # Recalculate cropped mask boundaries based on nonzero values (for internal use)
        XPixels_cropped = np.sum(Mask_cropped,axis=0)
        YPixels_cropped = np.sum(Mask_cropped,axis=1)
        p0_cropped = np.sum(np.cumsum(XPixels_cropped) == 0)
        p1_cropped = np.sum((np.cumsum(XPixels_cropped) - np.sum(XPixels_cropped)) > 0)
        l0_cropped = np.sum(np.cumsum(YPixels_cropped) == 0)
        l1_cropped = np.sum((np.cumsum(YPixels_cropped) - np.sum(YPixels_cropped)) > 0)

        # Define anterior boundary: 15% from the left edge of the mask
        ant_edge = p0_cropped+int(np.round((p1-p0)*0.15))

        # Crop image and mask to anterior segment only
        Mask_ant = Mask_cropped[l0_cropped:(l1_cropped+1),p0_cropped:(ant_edge+1)]
        Im1_ant = Img_cropped[l0_cropped:(l1_cropped+1),p0_cropped:(ant_edge+1),0]
        Im2_ant = Img_cropped[l0_cropped:(l1_cropped+1),p0_cropped:(ant_edge+1),1]
        Im3_ant = Img_cropped[l0_cropped:(l1_cropped+1),p0_cropped:(ant_edge+1),2]
        if (Img.shape[2] == 4):
            Im4_ant = Img_cropped[l0_cropped:(l1_cropped+1),p0_cropped:(ant_edge+1),3]

        #-------------------------#    
        # Optional: display cropped anterior segment across all channels
        # fig,ax = plt.subplots(ncols=Img.shape[2]+1,figsize=(9,3),sharey=True)
        # ax[0].imshow(Mask_ant,cmap='binary')
        # ax[1].imshow(Im1_ant)
        # ax[2].imshow(Im2_ant)
        # ax[3].imshow(Im3_ant)
        # if (Img.shape[2] == 4):
        #     ax[4].imshow(Im4_ant)
        # plt.show()
        #-------------------------#

        # Extract pixel intensities within anterior mask and compute z-scores using full-image stats
        z1_ant,z2_ant,z3_ant = Mask_ant*((Im1_ant - mu1)/sig1), Mask_ant*((Im2_ant - mu2)/sig2), Mask_ant*((Im3_ant - mu3)/sig3)
        if (Img.shape[2] == 4):
            z4_ant = Mask_ant*((Im4_ant - mu4)/sig4)

        # Initialise output DataFrame and result variables
        df_row = pd.DataFrame()
        ant_int_2 = np.sum(z2_ant)     # integrated enrichment for channel 2
        ant_int_3 = np.sum(z3_ant)     # integrated enrichment for channel 3
        corr_23 = np.nan               # initialise Pearson correlation ch2 vs ch3
        corr_24 = np.nan               # initialise Pearson correlation ch2 vs ch4
        ant_int_4 = np.nan             # initialise integrated enrichment for channel 4
        if (Img.shape[2] == 4):
            ant_int_4 = np.sum(z4_ant) # integrated enrichment for channel 4
        
        # Define intensity threshold in channel 2 (e.g., HA) using log-mean + 2*std
        y2_log=np.log10(y2[y2>0])
        y2_log_mean, y2_log_std = np.mean(y2_log), np.std(y2_log)
        thres=10**(y2_log_mean+2*y2_log_std)
        highHA_ant_mask = np.where(Im2_ant>thres,1,0)

        #-------------------------#
        # Optional: plot log10 histogram and intensity threshold
        # plt.hist(y2_log, bins=100, alpha=0.5)
        # plt.hist(y2_ant_log, bins=100, color='r', alpha=0.5)
        # plt.axvline(y2_log_mean+2*y2_log_std)
        # plt.show()
        #-------------------------#

        # Smooth masked images using Gaussian filter, then extract only high-HA anterior pixels
        y1_ant_smoothed,y2_ant_smoothed,y3_ant_smoothed = (ndimage.gaussian_filter(Im1_ant*Mask_ant, sigma=(5, 5)))[highHA_ant_mask > 0],(ndimage.gaussian_filter(Im2_ant*Mask_ant, sigma=(5, 5)))[highHA_ant_mask > 0],(ndimage.gaussian_filter(Im3_ant*Mask_ant, sigma=(5, 5)))[highHA_ant_mask > 0]
        if (Img.shape[2] == 4):
            y4_ant_smoothed = (ndimage.gaussian_filter(Im4_ant*Mask_ant, sigma=(5, 5)))[highHA_ant_mask > 0]
    
        # Compute Pearson correlation coefficients for HA (channel 2) vs target channels
        if (len(y2_ant_smoothed) > 2 and len(y3_ant_smoothed) > 2):
            corr_23=st.pearsonr(y2_ant_smoothed,y3_ant_smoothed)[0]
        if (Img.shape[2] == 4 and len(y2_ant_smoothed) > 2 and len(y4_ant_smoothed) > 2):
            corr_24=st.pearsonr(y2_ant_smoothed,y4_ant_smoothed)[0]
    
        #-------------------------#
        # Optional: display anterior smoothed maps and masked overlay
        # fig,ax = plt.subplots(ncols=Img.shape[2]+1,figsize=(9,3),sharey=True)
        # ax[0].imshow(highHA_ant_mask*Mask_ant,cmap='binary')
        # ax[1].imshow((ndimage.gaussian_filter(Im1_ant*Mask_ant, sigma=(5, 5)))*highHA_ant_mask)
        # ax[2].imshow((ndimage.gaussian_filter(Im2_ant*Mask_ant, sigma=(5, 5)))*highHA_ant_mask)
        # ax[3].imshow((ndimage.gaussian_filter(Im3_ant*Mask_ant, sigma=(5, 5)))*highHA_ant_mask)
        # ax[3].title.set_text(str(np.round(corr_23,2)))
        # if (Img.shape[2] == 4):
        #     ax[4].imshow((ndimage.gaussian_filter(Im4_ant*Mask_ant, sigma=(5, 5)))*highHA_ant_mask)
        #     ax[4].title.set_text(str(np.round(corr_24,2)))
        # plt.show()
        #-------------------------#

        # Combine results into a single-row DataFrame
        df_row = pd.DataFrame([[ant_int_2,ant_int_3,ant_int_4,corr_23,corr_24]], index=[file_prefix])    
        df_row.columns = ['Ant Int ch 2', 'Ant Int ch 3', 'Ant Int ch 4', 'Pears 2-3', 'Pears 2-4']
        return(df_row)

#............................................................................#
#............................................................................#
#............................................................................#

# Function to compute summary statistics and confidence intervals for a filtered subset of data
# Inputs:
# - ddf_working: pandas DataFrame containing numeric measurements, indexed by sample identifiers
# - specie: string used to filter rows (e.g. condition or genotype keyword present in the index)
# Returns:
# - A tuple of column-wise summary statistics:
#     - means: arithmetic mean
#     - ci_lowers: lower bound of 95% confidence interval (mean - 1.96 × SEM)
#     - ci_uppers: upper bound of 95% confidence interval (mean + 1.96 × SEM)
#     - stds: standard deviation
#     - counts: number of non-NaN values
#     - fourths: maximum values (used in plotting)
#     - medians: median values
#     - firsts: 25th percentiles (Q1)
#     - thirds: 75th percentiles (Q3)
#     - nineties: 90th percentiles (optional for whiskers or shading)
def return_mean_cis(ddf_working, specie):
    # Filter the DataFrame for rows whose index contains the given string 'specie'
    TF_array = [specie in ind for ind in ddf_working.index]
    ddf_specie = ddf_working[TF_array]

    # Compute basic statistics
    means = ddf_specie.mean()
    stds = ddf_specie.std()
    counts = ddf_specie.count()

    # Compute 95% confidence intervals based on normal approximation
    cis = 1.96 * stds / np.sqrt(counts)
    ci_lowers = means - cis
    ci_uppers = means + cis

    # Compute additional descriptive statistics
    fourths = np.max(ddf_specie)        # maximum value in each column
    medians = ddf_specie.median()       # median value
    firsts = ddf_specie.quantile(0.25)  # 25th percentile
    thirds = ddf_specie.quantile(0.75)  # 75th percentile
    nineties = ddf_specie.quantile(0.9) # 90th percentile

    # --------------------------#
    # Optional: use ±1 standard deviation instead of confidence intervals
    # ci_lowers = means - stds
    # ci_uppers = means + stds
    # --------------------------#

    return(means, ci_lowers, ci_uppers, stds, counts, fourths, medians, firsts, thirds, nineties)

#............................................................................#
#............................................................................#
#............................................................................#

# Function to plot average z-score linescans across the anterior-posterior axis for multiple genotypes
# Inputs:
# - species_plot: list of genotype identifiers to plot
# - labels: display labels corresponding to each genotype
# - colours: list of colours for each genotype trace
# - channel: string used for labelling (e.g. 'Vasa', 'nanos', 'pgc')
# - ddf: DataFrame containing z-score linescans (indexed by sample)
# - date: date string for file output
# - cell: string for sample type (e.g. 'oocyte', 'embryo')
# - tissue_stain: label indicating the stained marker (used in filenames)
# - ymin, ymax: axis limits for plots
# - PS: if True, save figure to file
# - form: file format to save ('svg' or 'png')
# - extra: optional row name in ddf to highlight as an overlay
def return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=False, form="svg", extra = np.nan):
   
    n_conditions = len(species_plot)
    font_size = 6
    tick_width = 0.5
    tick_size = 2
    line_width = 0.5
    font = FontProperties()
    font.set_name('Arial')
    
    # Initialise figure with two subplots per row (e.g., HA and second marker)
    fig, ax = plt.subplots(nrows=n_conditions, ncols=2, figsize= (2*4, len(species_plot)*2),sharex='col',sharey='row')

    for specie_i in range(len(species_plot)):
        
        # Plot control: Oregon R
        specie_plottables = return_mean_cis(ddf, 'OR')
        ax[specie_i, 0].plot(specie_plottables[0][0:300], specie_plottables[0][300:600], color='gray', linewidth=line_width, alpha=1)
        ax[specie_i, 0].fill_between(
            list(specie_plottables[0][0:300]), specie_plottables[1][300:600], specie_plottables[2][300:600], color='gray', alpha=.1)
        ax[specie_i, 0].set_ylim(ymin,ymax)
        ax[specie_i, 1].plot(specie_plottables[0][0:300], specie_plottables[0][600:900], color='gray', linewidth=line_width, alpha=1)
        ax[specie_i, 1].fill_between(
            list(specie_plottables[0][0:300]), specie_plottables[1][600:900], specie_plottables[2][600:900], color='gray', alpha=.1)
        ax[specie_i, 1].set_ylim(ymin,ymax)
        OR_count = specie_plottables[4][0]

        # Plot positive control: WT oskar (84x34)
        specie_plottables = return_mean_cis(ddf, '84x34')
        ax[specie_i, 0].plot(specie_plottables[0][0:300], specie_plottables[0][300:600], color='k', linewidth=line_width, alpha=1)
        ax[specie_i, 0].fill_between(
            list(specie_plottables[0][0:300]), specie_plottables[1][300:600], specie_plottables[2][300:600], color='k', alpha=.1)
        ax[specie_i, 0].set_ylim(ymin,ymax)
        ax[specie_i, 1].plot(specie_plottables[0][0:300], specie_plottables[0][600:900], color='k', linewidth=line_width, alpha=1)
        ax[specie_i, 1].fill_between(
            list(specie_plottables[0][0:300]), specie_plottables[1][600:900], specie_plottables[2][600:900], color='k', alpha=.1)
        ax[specie_i, 1].set_ylim(ymin,ymax)
        WTosk_count = specie_plottables[4][0]

        # Optional: overlay a specific individual trace
        if not pd.isna(extra):
            ax[specie_i, 0].plot(specie_plottables[0][0:300], ddf.loc[extra][300:600], color='#b3a735ff', linewidth=line_width*2, alpha=1)
            ax[specie_i, 1].plot(specie_plottables[0][0:300], ddf.loc[extra][600:900], color='#b3a735ff', linewidth=line_width*2, alpha=1)

        # Plot experimental mutant
        specie_plottables = return_mean_cis(ddf, species_plot[specie_i])
        ax[specie_i, 0].plot(specie_plottables[0][0:300], specie_plottables[0][300:600], color=colours[specie_i], linewidth=line_width, alpha=1)
        ax[specie_i, 0].fill_between(
            list(specie_plottables[0][0:300]), specie_plottables[1][300:600], specie_plottables[2][300:600], color=colours[specie_i], alpha=.1)
        ax[specie_i, 0].set_ylim(ymin,ymax)
        ax[specie_i, 1].plot(specie_plottables[0][0:300], specie_plottables[0][600:900], color=colours[specie_i], linewidth=line_width, alpha=1)
        ax[specie_i, 1].fill_between(
            list(specie_plottables[0][0:300]), specie_plottables[1][600:900], specie_plottables[2][600:900], color=colours[specie_i], alpha=.1)
        ax[specie_i, 1].set_ylim(ymin,ymax)
        mut_label = labels[specie_i]
        # ax[specie_i, 0].text(0.25, 0.9, mut_label+" (N="+str(specie_plottables[4][0])+")", horizontalalignment='left', verticalalignment='center', transform=ax[specie_i, 0].transAxes, color=colours[specie_i], fontsize = font_size)
        # ax[specie_i, 1].text(1.35, 0.9, mut_label+" (N="+str(specie_plottables[4][0])+")", horizontalalignment='left', verticalalignment='center', transform=ax[specie_i, 0].transAxes, color=colours[specie_i], fontsize = font_size)


        # --------------------------#
        # Optional: add titles and sample size annotations
        # if (specie_i == 0):
        #     ax[specie_i, 0].set_title('HA', fontsize = font_size)
        #     if (channel == "nanos"):
        #         ax[specie_i, 1].set_title('$\it{nanos}$', fontsize = font_size)
        #     if (channel == "pgc"):
        #         ax[specie_i, 1].set_title('$\it{pgc}$', fontsize = font_size)
        #     else:
        #         ax[specie_i, 1].set_title(channel, fontsize = font_size)
        #     ax[specie_i, 0].text(0.25, 0.8, "WT Oskar (N="+str(WTosk_count)+")", horizontalalignment='left', verticalalignment='center', transform=ax[specie_i, 0].transAxes, color='k', fontsize = font_size)
        #     ax[specie_i, 1].text(1.35, 0.8, "WT Oskar (N="+str(WTosk_count)+")", horizontalalignment='left', verticalalignment='center', transform=ax[specie_i, 0].transAxes, color='k', fontsize = font_size)
        #     ax[specie_i, 0].text(0.25, 0.7, "OR (N="+str(OR_count)+")", horizontalalignment='left', verticalalignment='center', transform=ax[specie_i, 0].transAxes, color='gray', fontsize = font_size)
        #     ax[specie_i, 1].text(1.35, 0.7, "OR (N="+str(OR_count)+")", horizontalalignment='left', verticalalignment='center', transform=ax[specie_i, 0].transAxes, color='gray', fontsize = font_size)
        # --------------------------#

        # Format ticks and axes
        ax[specie_i,0].set_ylim(ymin,ymax)
        ax[specie_i,0].yaxis.set_ticks([int(ymin), int((ymax+ymin)/2), int(ymax)])
        # ax[specie_i,0].yaxis.set_ticklabels([int(ymin),'',int(ymax)], size=font_size)
        ax[specie_i,0].yaxis.set_ticklabels(['','',''])
        ax[specie_i,0].tick_params(size=tick_size,width=tick_width)
        # ax[specie_i,0].xaxis.set_ticklabels(["",0,"",1], size=font_size)
        ax[specie_i,0].xaxis.set_ticklabels([""])
        ax[specie_i,1].set_ylim(ymin,ymax)
        ax[specie_i,1].tick_params(size=tick_size,width=tick_width)
        # ax[specie_i,1].yaxis.set_ticklabels([int(ymin),'',int(ymax)], size=font_size)
        ax[specie_i,1].xaxis.set_ticklabels([""])
        # ax[specie_i,1].xaxis.set_ticklabels(["",0,"",1], size=font_size)
        # ax[n_conditions-1,0].set_xlabel("Position", size=60)
        # ax[n_conditions-1,1].set_xlabel("Position", size=60)

    # Save output if requested
    if (PS == True):
        if (form == "png"):
            plt.savefig("directory\\linescans"+date+"_"+cell+"_"+tissue_stain+"_HA-"+channel+"_ZscoreOverPosition_linescan.png",format='png', dpi=300)
        if (form == "svg"):
            plt.savefig('directory\\linescans'+date+"_"+cell+"_"+tissue_stain+"_HA-"+channel+"_ZscoreOverPosition_linescan.svg", format='svg', dpi=300)

#............................................................................#
#............................................................................#
#............................................................................#

# Function to estimate a p-value for the difference in means between two distributions using bootstrap resampling
# Inputs:
# - x1: first 1D array-like distribution (e.g., test condition)
# - x2: second 1D array-like distribution (e.g., control condition)
# Returns:
# - p_val: estimated one-tailed p-value indicating whether mean(x1) > mean(x2)
#   (or vice versa, depending on direction of observed effect)
def myBootstrap_pVal(x1, x2):

    # Set number of bootstrap replicates
    n_reps = 10000
    n1 = len(x1)
    n2 = len(x2)
    
    # Observed test statistic: difference in means
    sample_mean = np.mean(x1)-np.mean(x2)

    # Allocate array to store bootstrap replicate statistics
    bootstrap_mean = np.zeros(n_reps);

    # Generate bootstrap replicates by sampling with replacement
    for i in range(n_reps):
        sample_X1 = x1[np.int_(np.random.rand(n1,1)*n1)]
        sample_X2 = x2[np.int_(np.random.rand(n2,1)*n2)]
        bootstrap_mean[i] = np.mean(sample_X1)-np.mean(sample_X2)

    # Estimate bootstrap distribution parameters
    boot_mean = np.mean(bootstrap_mean)
    boot_std = np.std(bootstrap_mean)
    
    # Evaluate cumulative distribution function over a broad range
    x_data = np.arange(-100, 100, 1)
    y_cdf_data = norm.cdf(x_data, boot_mean, boot_std)

    # Compute one-tailed p-value based on observed statistic
    if sample_mean == 0:
        p_val = 'NaN'
    if sample_mean > 0:
        p_val = y_cdf_data[x_data==0]
    if sample_mean < 0:
        p_val = 1 - y_cdf_data[x_data==0]
    return(p_val)

#............................................................................#
#............................................................................#
#............................................................................#

# Function to generate annotated horizontal boxplots for integrated enrichment or correlation metrics
# Inputs:
# - species_plot: list of genotype identifiers to include
# - labels_plus: corresponding display labels for each genotype
# - coloursPlus: list of colours for boxplots, matched to genotypes
# - ddf_filtered: DataFrame containing measurement values (indexed by sample name)
# - cell: label indicating tissue or stage (e.g., 'oocyte', 'embryo')
# - stain: name of the molecular stain (e.g., 'Vasa', 'pgc') for file naming
# - date: string used in output file names
# - xlimits: tuple of (xmin, xmax) for x-axis range
# - pos_ctrl: genotype identifier for positive control (default: '84x34')
# - neg_ctrl: optional genotype identifier for negative control (default: NaN)
# Returns:
# - Tuple of lists: (p-values vs neg_ctrl, p-values vs pos_ctrl)
def return_box(species_plot, labels_plus, coloursPlus, ddf_filtered, cell, stain, date, xlimits, pos_ctrl = '84x34', neg_ctrl = np.nan):

    # Extract values for negative control (if provided)
    if pd.notnull(neg_ctrl):
        TF_array = [neg_ctrl in ind for ind in ddf_filtered.index]
        ddf_neg = np.array(ddf_filtered[TF_array])
        med_neg = np.median(ddf_neg)
        pValues_Neg = [float("NaN")]*(len(species_plot))
        pValues_Neg_symbolic = [""]*(len(species_plot))
   
    # Extract values for positive control
    TF_array = [pos_ctrl in ind for ind in ddf_filtered.index]
    ddf_pos = np.array(ddf_filtered[TF_array])
    med_pos = np.median(ddf_pos)
    pValues_Pos = [float("NaN")]*(len(species_plot))
    pValues_Pos_symbolic = [""]*(len(species_plot))

    # Gather data per genotype and perform bootstrap significance tests
    ddf = []
    for k in list(range(len(species_plot))):
        TF_array = [species_plot[k] in ind for ind in ddf_filtered.index]
        ddf_specie = np.array(ddf_filtered[TF_array])
        ddf.append(ddf_specie)
        if (len(ddf_specie) != 0):
            # Compare to positive control
            pval_specie = myBootstrap_pVal(ddf_pos,ddf_specie)
            pValues_Pos[k] = pval_specie    # Optional: use Bonferroni correction → pval * len(species_plot)
                                            # Currently not applied but left available for future use
            # Compare to negative control (if defined)
            if pd.notnull(neg_ctrl):
                pval_specie = myBootstrap_pVal(ddf_neg,ddf_specie)
                pValues_Neg[k] = pval_specie    # Optional: use Bonferroni correction → pval * len(species_plot)
                                                # Currently not applied but left available for future use

    # Manually set reference p-values to NaN (self-comparisons)
    pValues_Pos[-1] = 'NaN'
    if pd.notnull(neg_ctrl):
        pValues_Neg[0] = 'NaN'

    # Convert p-values into symbolic notation
    for i in range(len(species_plot)):
        if (pValues_Pos[i] != "NaN"):
            if (pValues_Pos[i] < 0.05):
                pValues_Pos_symbolic[i] = "*"
        if pd.notnull(neg_ctrl):
            if (pValues_Neg[i] != "NaN"):
                if (pValues_Neg[i] < 0.05):
                    pValues_Neg_symbolic[i] = "$\u2021$"

    # Print per-genotype summary to console
    # for i in range(len(species_plot)):
    #     print('Specie = ',species_plot[i],
    #           'Label = ',labels_plus[i],
    #           'N =',len(ddf_filtered[ddf_filtered.index.str.contains(species_plot[i], regex=True)]),
    #           'p-value pos = ',pValues_Pos[i],
    #             'p-value neg = ',pValues_Neg[i]
    #           )
    for i in range(len(species_plot)):
        print('Specie = ',species_plot[i],
              'Label = ',labels_plus[i],
              'N =',len(ddf_filtered[ddf_filtered.index.str.contains(species_plot[i], regex=True)]),
              'p-value pos = ',pValues_Pos[i]
              )

    # --------------------------#
    # Generate horizontal boxplot with overlaid scatter and symbolic annotation
    textsize = 30
    fig, ax = plt.subplots(figsize= (10, len(species_plot)*2)) #initialise plot

    # Core boxplot
    bp = ax.boxplot(ddf, patch_artist = True, notch ='True', vert = 0, showfliers=False, widths=0.6)
    ax.invert_yaxis()

    # Custom box styling
    for patch, colour in zip(bp['boxes'], coloursPlus):
        patch.set_facecolor(colour)
        patch.set_alpha(0.4)
    for whisker in bp['whiskers']:
        whisker.set(color ='grey', linewidth = 6, linestyle =":")
    for cap in bp['caps']:
        cap.set(color ='grey', linewidth = 6)
    for median, colour in zip(bp['medians'], coloursPlus):
        median.set(color = 'k', linewidth = 6)

    # Y-axis tick labels (mutant labels)
    ax.set_yticklabels(labels_plus)
    ax.tick_params(size=12,width=3)    

    # Overlay individual data points with jitter
    jitter = 0.04
    y_data = [np.array([i+1] * len(d)) for i, d in enumerate(ddf)]
    y_data_jittered = [y + st.t(df=6, scale=jitter).rvs(len(y)) for y in y_data]
    for x, y, colour in zip(ddf, y_data_jittered, coloursPlus):
        ax.scatter(x, y, marker='.', color='k')

    # Set x-axis limits and tick positions
    ax.set_xlim(xlimits)
    xmin, xmax = np.round(ax.get_xlim())

    # Add symbolic p-value annotations and mutant labels
    for c in range(len(labels_plus)):
        plt.text(xmax+((xmax-xmin)/10), c+1, pValues_Pos_symbolic[c], fontsize = textsize, color = "black")
        if pd.notnull(neg_ctrl):
            plt.text(xmax, c+1, pValues_Neg_symbolic[c], fontsize = textsize, color="grey")
        plt.text(xmin-(2*(xmax-xmin)/10), c+1, labels_plus[c], fontsize = textsize, color = "black")

    ax.xaxis.set_ticks([xmin, 0, xmax])
    ax.xaxis.set_ticklabels([xmin,0,xmax], size=textsize)
    ax.tick_params(size=15,width=7)
    # Remove y-axis ticks (labels are manually placed)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])

    # Reference lines for control medians
    if pd.notnull(neg_ctrl):
        ax.axvline(x = med_neg, color = 'gray', label = 'axvline - full height')
    ax.axvline(x = med_pos, color = 'black', label = 'axvline - full height')

    # Title and save figure
    plt.title(cell+stain, fontsize = textsize)
    plt.savefig("directory\\"+date+"_boxplots"+cell+stain+".svg", format='svg',dpi=300)
    return(pValues_Neg, pValues_Pos)

#............................................................................#
#............................................................................#
#............................................................................#

# Function to generate a compact, publication-quality horizontal boxplot figure without annotations
# Inputs:
# - species_plot: list of genotype identifiers to plot
# - labels_plus: list of display labels for each genotype
# - coloursPlus: list of colours for each box, matched to genotypes
# - ddf_filtered: DataFrame containing measurement values (indexed by sample name)
# - cell: label for tissue or sample type (e.g., 'oocyte', 'embryo')
# - stain: name of the molecular stain (e.g., 'Vasa', 'pgc') for file naming
# - date: string for output file naming
# - xlimits: tuple specifying x-axis limits (xmin, xmax)
# - pos_ctrl: identifier for positive control condition (default: '84x34')
# - neg_ctrl: optional identifier for negative control (default: NaN)
# Returns:
# - None (figure is saved directly to file)
def return_box_figure(species_plot, labels_plus, coloursPlus, ddf_filtered, cell, stain, date, xlimits, pos_ctrl = '84x34', neg_ctrl = np.nan):

    # Extract values for negative control (optional)
    if pd.notnull(neg_ctrl):
        TF_array = [neg_ctrl in ind for ind in ddf_filtered.index]
        ddf_neg = np.array(ddf_filtered[TF_array])
        med_neg = np.median(ddf_neg)
        pValues_Neg = [float("NaN")]*(len(species_plot))
        pValues_Neg_symbolic = [""]*(len(species_plot))
   
    # Extract values for positive control
    TF_array = [pos_ctrl in ind for ind in ddf_filtered.index]
    ddf_pos = np.array(ddf_filtered[TF_array])
    med_pos = np.median(ddf_pos)
    pValues_Pos = [float("NaN")]*(len(species_plot))
    pValues_Pos_symbolic = [""]*(len(species_plot))

    # Extract data per genotype into a list of arrays
    ddf = []
    for k in list(range(len(species_plot))):
        TF_array = [species_plot[k] in ind for ind in ddf_filtered.index]
        ddf_specie = np.array(ddf_filtered[TF_array])
        ddf.append(ddf_specie)

    # --------------------------#
    # Initialise compact figure with journal-ready settings
    width = 1.4694    # in inches (approx. 105.8 pt for single column)
    height = 3.3952   # in inches (approx. 244.5 pt to fit long genotype list)
    tick_length = 3   # tick size in points
    tick_width = 0.5  # tick line thickness
    line_width = 0.5  # boxplot line thickness
    point_size = 0.25 # scatter point area (diameter squared, in pt)
    
    fig, ax = plt.subplots(figsize= (width , height), constrained_layout=True) #initialise plot
    
    # Generate horizontal boxplot
    bp = ax.boxplot(ddf, boxprops=dict(linewidth=line_width), patch_artist = True, notch ='True', vert = 0, showfliers=False, widths=0.6)
    ax.invert_yaxis()
    
    # Apply custom box styling
    for patch, colour in zip(bp['boxes'], coloursPlus):
        patch.set_facecolor(colour)
        patch.set_alpha(0.4)
    for whisker in bp['whiskers']:
        whisker.set(color ='grey', linewidth = line_width, linestyle =":")
    for cap in bp['caps']:
        cap.set(color ='grey', linewidth = line_width)
    for median, colour in zip(bp['medians'], coloursPlus):
        median.set(color = 'k', linewidth = line_width)

    # --------------------------#
    # Optional: label y-axis ticks (suppressed for minimal figure style)
    # ax.set_yticklabels(labels_plus)
    # --------------------------#
    ax.tick_params(size=tick_length,width=tick_width)    

    # Add individual data points with jitter
    jitter = 0.04 
    y_data = [np.array([i+1] * len(d)) for i, d in enumerate(ddf)] 
    y_data_jittered = [y + st.t(df=6, scale=jitter).rvs(len(y)) for y in y_data]
    for x, y, colour in zip(ddf, y_data_jittered, coloursPlus):
        ax.scatter(x, y, marker='.', s=point_size, color='k')

    # Set axis limits and ticks
    ax.set_xlim(xlimits)
    # xmin, xmax = np.round(ax.get_xlim())
    # for c in range(len(labels_plus)):
    #     plt.text(xmax+((xmax-xmin)/10), c+1, pValues_Pos_symbolic[c], fontsize = textsize, color = "black")
    #     if pd.notnull(neg_ctrl):
    #         plt.text(xmax, c+1, pValues_Neg_symbolic[c], fontsize = textsize, color="grey")
    #     plt.text(xmin-(2*(xmax-xmin)/10), c+1, labels_plus[c], fontsize = textsize, color = "black")
    ax.xaxis.set_ticks([xlimits[0], 0, xlimits[1]])
    # ax.xaxis.set_ticklabels([xlimits[0], 0, xlimits[1]], size=textsize)
    ax.tick_params(size=tick_length,width=tick_width)

    # Remove all tick labels to maintain minimal, clean formatting
    # ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([])

    # --------------------------#
    # Optional: add vertical reference lines for control medians
    # if pd.notnull(neg_ctrl):
    #     ax.axvline(x=med_neg, color='gray')
    # ax.axvline(x=med_pos, color='black')
    # --------------------------#
    
    # Save the figure to file
    # plt.title(cell+stain, fontsize = textsize)
    plt.savefig("directory\\"+date+"_boxplots4fig"+cell+stain+".svg", format='svg',dpi=300)


#============================================================================#
#============================================================================#
#============================================================================#

# -----------------------------------------------------------------------------
# ANALYSIS: Extract spatial and enrichment metrics from micrographs
# -----------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Section 1: Process Vasa-stained oocytes (adov_HAVasa)
# --------------------------------------------------------------------------


# Define input directory for rotated, aligned micrographs
sample_dir = "adov_HAVasa\\"

# Identify all unique image filenames (excluding extensions)
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]

# Initialise output DataFrames for this dataset
ddf_oocytes_HAVasa = pd.DataFrame()
ddf_oocytes_HAVasa_linescans = pd.DataFrame()

# Loop over all micrographs and extract both linescan and anterior metrics
for prefix in UnqFiles_prefixes:
    print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_oocytes_HAVasa_linescans = pd.concat([ddf_oocytes_HAVasa_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_oocytes_HAVasa = pd.concat([ddf_oocytes_HAVasa,df])

# Prepare z-score linescan data for plotting (HA and Vasa channels)
ddf = pd.concat([
    ddf_oocytes_HAVasa_linescans.iloc[:, 0:300],
    ddf_oocytes_HAVasa_linescans.iloc[:, 600:1200]
], axis=1)
ddf.columns = range(0, 900)

# Define plot configuration
date = "25-4-8"
channel = 'Vasa'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "all_adov"
tissue_stain = 'Vasa'
ymin=-2
ymax=7
# Generate linescan plots
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg")

# Split anterior metric outputs into separate data series
ddf_oocytes_HAVasa_AntInt_HA = ddf_oocytes_HAVasa['Ant Int ch 2'].dropna()
ddf_oocytes_HAVasa_AntInt_Vasa = ddf_oocytes_HAVasa['Ant Int ch 3'].dropna()
ddf_oocytes_HAVasa_Pears_HAVasa = ddf_oocytes_HAVasa['Pears 2-3'].dropna()

# --------------------------------------------------------------------------
# Section 2: Process early embryo datasets (HA-Vasa, HA-nanos, HA-pgc)
# --------------------------------------------------------------------------

# Repeat for HA-Vasa embryos
sample_dir = "earlyBros_HAVasa\\"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HAVasa = pd.DataFrame()
ddf_embryos_HAVasa_linescans = pd.DataFrame()
for prefix in UnqFiles_prefixes:
    print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_HAVasa_linescans = pd.concat([ddf_embryos_HAVasa_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_embryos_HAVasa = pd.concat([ddf_embryos_HAVasa,df])
# Extract relevant metrics (HA = ch2, Vasa = ch4, Pearson = 2-4)
ddf_embryos_HAVasa_AntInt_HA = ddf_embryos_HAVasa['Ant Int ch 2'].dropna()
ddf_embryos_HAVasa_AntInt_Vasa = ddf_embryos_HAVasa['Ant Int ch 4'].dropna()
ddf_embryos_HAVasa_Pears_HAVasa = ddf_embryos_HAVasa['Pears 2-4'].dropna()
#............................................................................#
# Repeat for HA-nanos embryos
sample_dir = "earlyBros_HAnanos\\"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HAnanos = pd.DataFrame()
ddf_embryos_HAnanospgc_linescans = pd.DataFrame()
for prefix in UnqFiles_prefixes:
    print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_HAnanospgc_linescans = pd.concat([ddf_embryos_HAnanospgc_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_embryos_HAnanos = pd.concat([ddf_embryos_HAnanos,df])
ddf_embryos_HAnanos_AntInt_HA = ddf_embryos_HAnanos['Ant Int ch 2'].dropna()
ddf_embryos_HAnanos_AntInt_nanos = ddf_embryos_HAnanos['Ant Int ch 3'].dropna()
ddf_embryos_HAnanos_Pears_HAnanos = ddf_embryos_HAnanos['Pears 2-3'].dropna()
#............................................................................#
# Repeat for HA-pgc embryos
sample_dir = "earlyBros_HApgc\\"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HApgc = pd.DataFrame()
ddf_embryos_HAnanospgc_linescans = pd.DataFrame()
for prefix in UnqFiles_prefixes:
    print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_HAnanospgc_linescans = pd.concat([ddf_embryos_HAnanospgc_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_embryos_HApgc = pd.concat([ddf_embryos_HApgc,df])
ddf_embryos_HApgc_AntInt_HA = ddf_embryos_HApgc['Ant Int ch 2'].dropna()
ddf_embryos_HApgc_AntInt_pgc = ddf_embryos_HApgc['Ant Int ch 4'].dropna()
ddf_embryos_HApgc_Pears_HApgc = ddf_embryos_HApgc['Pears 2-4'].dropna()

# --------------------------------------------------------------------------
# Section 3: Generate linescan plots for embryos (Vasa, nanos, pgc)
# --------------------------------------------------------------------------

# Vasa (embryos)
ddf = pd.concat([
    ddf_embryos_HAVasa_linescans.iloc[:, 0:300],
    ddf_embryos_HAVasa_linescans.iloc[:, 600:900],
    ddf_embryos_HAVasa_linescans.iloc[:, 1200:1500]
], axis=1)
ddf.columns = range(0, 900)
date = "25-4-8"
channel = 'Vasa'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "all_embryos"
tissue_stain = 'Vasa'
ymin=-2
ymax=7
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg")
#............................................................................#
# nanos (embryos)
ddf = pd.concat([
    ddf_embryos_HAnanospgc_linescans.iloc[:, 0:300],
    ddf_embryos_HAnanospgc_linescans.iloc[:, 600:900],
    ddf_embryos_HAnanospgc_linescans.iloc[:, 900:1200]
], axis=1)
ddf.columns = range(0, 900)
date = "25-4-8"
channel = 'nanos'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "all_embryos"
tissue_stain = 'nanos_extra'
ymin=-2
ymax=15
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg", extra='84x34_m22-12-25_sl2_ser34')
#............................................................................#
# pgc (embryos)
ddf = pd.concat([
    ddf_embryos_HAnanospgc_linescans.iloc[:, 0:300],
    ddf_embryos_HAnanospgc_linescans.iloc[:, 600:900],
    ddf_embryos_HAnanospgc_linescans.iloc[:, 1200:1500]
], axis=1)
ddf.columns = range(0, 900)
date = "25-4-8"
channel = 'pgc'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "all_embryos"
tissue_stain = 'pgc_extra'
ymin=-2
ymax=15
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg", extra='84x34_m22-12-25_sl2_ser34')

# --------------------------------------------------------------------------
# Section 4: Plot integrated anterior enrichment values
# --------------------------------------------------------------------------

date_today = '25-03-14'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['gray']+['white']*14+['k']
cond_cell = "oocyte"
Vasa_adov_xlimits = [-75000, 200000]
stain_cell = 'Vasa_ant_int'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_adov_xlimits, neg_ctrl = 'OR')
# return_box_figure(species_plot, labels, colours, ddf_oocytes_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_adov_xlimits, neg_ctrl = 'OR')
#............................................................................#
cond_cell = "bros"
stain_cell = 'Vasa_ant_int'
Vasa_bros_xlimits = [-265000, 295000]
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_bros_xlimits, neg_ctrl = 'OR')
# return_box_figure(species_plot, labels, colours, ddf_embryos_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_bros_xlimits, neg_ctrl = 'OR')
#............................................................................#
stain_cell = 'nanos_ant_int'
nanos_bros_xlimits = [-140000, 220000]
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_AntInt_nanos, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=nanos_bros_xlimits, neg_ctrl = 'OR')
# return_box_figure(species_plot, labels, colours, ddf_embryos_HAnanos_AntInt_nanos, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=nanos_bros_xlimits, neg_ctrl = 'OR')
#............................................................................#
stain_cell = 'pgc_ant_int'
pgc_bros_xlimits = [-149000, 165000]
return_box(species_plot, labels, colours, ddf_embryos_HApgc_AntInt_pgc, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=pgc_bros_xlimits, neg_ctrl = 'OR')
# return_box_figure(species_plot, labels, colours, ddf_embryos_HApgc_AntInt_pgc, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=pgc_bros_xlimits, neg_ctrl = 'OR')

# --------------------------------------------------------------------------
# Section 5: Plot Pearson’s correlation coefficients
# --------------------------------------------------------------------------

species_plot = ['84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['white']*14+['k']
cond_cell = "oocyte"
stain_cell = 'HA_Vasa_pears'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_Pears_HAVasa, cell=cond_cell, stain=stain_cell,date=date_today,xlimits=[-1, 1])
cond_cell = "embryo"
stain_cell = 'HA_Vasa_pears'
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_Pears_HAVasa, cell=cond_cell, stain=stain_cell,date=date_today,xlimits=[-1, 1])
cond_cell = "embryo"
stain_cell = 'HA_nanos_pears'
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_Pears_HAnanos, cell=cond_cell, stain=stain_cell,date=date_today,xlimits=[-1, 1])
stain_cell = 'HA_pgc_pears'
return_box(species_plot, labels, colours, ddf_embryos_HApgc_Pears_HApgc, cell=cond_cell, stain=stain_cell,date=date_today,xlimits=[-1, 1])


#============================================================================#
#============================================================================#
#============================================================================#

# -----------------------------------------------------------------------------
# ANALYSIS: Pole cell quantification and statistical testing (Figure 2)
# -----------------------------------------------------------------------------

# Load summary table of pole cell phenotypes (corresponding to Supplementary Table 2)
poleCellDF = pd.read_csv('directory\\lateBros_poleCells_data_summ_Jan24CSV.csv', index_col=0, float_precision='round_trip')

# Define plotting order and display labels
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']

# -----------------------------------------------------------------------------
# Section 1: Fisher's exact tests for frequency of anterior pole cells
# -----------------------------------------------------------------------------


from scipy.stats import fisher_exact
pval_data = []

# Define controls
oregonr_pos = poleCellDF.loc['OregonR', 'N anterior PCs']
oregonr_total = poleCellDF.loc['OregonR', 'N stage 5']
positive_pos = poleCellDF.loc['WT Oskar-HA-bcd3UTR', 'N anterior PCs']
positive_total = poleCellDF.loc['WT Oskar-HA-bcd3UTR', 'N HA+']

# Perform Fisher's exact test for each condition vs both controls
for condition, row in poleCellDF.iterrows():
    if condition in ['OregonR', 'WT Oskar-HA-bcd3UTR']:
        continue
    this_pos = row['N anterior PCs']
    this_total = row['N HA+'] if condition != 'OregonR' else row['N stage 5']

    #2x2 contingency tables
    table_vs_oregonr = [[this_pos, this_total - this_pos],
                        [oregonr_pos, oregonr_total - oregonr_pos]]
    table_vs_positive = [[this_pos, this_total - this_pos],
                         [positive_pos, positive_total - positive_pos]]

    #Fisher's exact test
    _, pval_vs_oregonr = fisher_exact(table_vs_oregonr, alternative='greater')
    _, pval_vs_positive = fisher_exact(table_vs_positive, alternative='less')

    pval_data.append({
        'Condition': condition,
        'p-value vs OregonR': pval_vs_oregonr,
        'p-value vs WT Oskar-HA-bcd3UTR': pval_vs_positive
    })
pvalDF = pd.DataFrame(pval_data).set_index('Condition')

# -----------------------------------------------------------------------------
# Section 2: Bootstrap comparisons of pole cell counts (anterior and posterior)
# -----------------------------------------------------------------------------

# Define parsing function for stringified arrays (comma-separated integers)
def parse_pc_string(s):
    if isinstance(s, str):
        s = s.strip()
        if s.lower() == 'nan' or s == '':
            return np.nan
        return np.array([int(x.strip()) for x in s.split(',') if x.strip().isdigit()])
    return np.nan

poleCellDF['Count posterior PCs'] = poleCellDF['Count posterior PCs'].apply(parse_pc_string)
poleCellDF['Count anterior PCs'] = poleCellDF['Count anterior PCs'].apply(parse_pc_string)

# Extract posterior and anterior control datasets
posterior_control = poleCellDF.loc['OregonR', 'Count posterior PCs']
anterior_control = poleCellDF.loc['WT Oskar-HA-bcd3UTR', 'Count anterior PCs']

# Apply bootstrap comparisons to all conditions
pval_pc_data = []
for condition, row in poleCellDF.iterrows():
    test_posterior = row['Count posterior PCs']
    test_anterior = row['Count anterior PCs']
    if not (isinstance(test_posterior, float) and np.isnan(test_posterior)):
        test_posterior = np.array(test_posterior)
        pval_posterior = myBootstrap_pVal(test_posterior, np.array(posterior_control))
    if not (isinstance(test_anterior, float) and np.isnan(test_anterior)):
        test_anterior = np.array(test_anterior)
        pval_anterior = myBootstrap_pVal(test_anterior, np.array(anterior_control))
    pval_pc_data.append({
        'Condition': condition,
        'p-value (posterior PCs vs OregonR)': pval_posterior,
        'p-value (anterior PCs vs WT Oskar-HA-bcd3UTR)': pval_anterior
    })

# -----------------------------------------------------------------------------
# Section 3: Plot stacked bar graph of pole cell and attempt frequencies
# -----------------------------------------------------------------------------

# Calculate per-genotype fractions
date = "date of figure generation"
poleCells_fraction = poleCellDF["N ant PCs/N (HA+)"]
poleCells_fraction['OregonR'] = poleCellDF["N ant PCs/N (all)"][0]
DevDefects_fraction = poleCellDF['N defect/N (HA+)']
DevDefects_fraction['OregonR'] = poleCellDF["N defect/N (all)"][0]
poleCellAttempts_fraction = poleCellDF['N PC attempt/N (HA+)']
poleCellAttempts_fraction['OregonR'] = poleCellDF["N PC attempt/N (all)"][0]
ind = np.arange(0, len(labels))

# Plot stacked bar for attempts (white) and confirmed pole cells (hatched)
fig, ax = plt.subplots(figsize= (6, len(species_plot)*0.6))
ax.barh(np.arange(0, len(labels)), poleCellAttempts_fraction, height=0.8, color='white', edgecolor='k', label = "Fraction with Anterior Pole Cell Attempts")
ax.barh(np.arange(0, len(labels)), poleCells_fraction, height=0.8, left=poleCellAttempts_fraction, color='white', edgecolor='k', hatch=r"///", label = "Fraction with Anterior Pole Cells")
ax.tick_params(size=10,width=3)
ax.set(yticks=np.arange(0, len(labels)), yticklabels=poleCellAttempts_fraction.index)
ax.invert_yaxis() 
ax.legend(loc = 'upper right')
plt.savefig("directory\\"+date+"_PoleCells_Attempts_Fractions_stackbar.svg", format='svg',dpi=300)

# -----------------------------------------------------------------------------
# Section 4: Plot scatter of individual anterior/posterior pole cell counts
# -----------------------------------------------------------------------------

# Convert stringified arrays to numeric arrays
ant_poleCells_count = poleCellDF["Count anterior PCs"]
ant_poleCells_count_int = ant_poleCells_count
for jj in range(len(ant_poleCells_count)):
    if type(ant_poleCells_count[jj]) == str:
        ant_poleCells_count_int[jj] = np.int_(ant_poleCells_count[jj].split(', '))
ant_poleCells_count_df = pd.DataFrame(ant_poleCells_count_int).dropna().transpose()

post_poleCells_count = poleCellDF["Count posterior PCs"]
post_poleCells_count_int = post_poleCells_count
for jj in range(len(post_poleCells_count)):
    if type(post_poleCells_count[jj]) == str:
        post_poleCells_count_int[jj] = np.int_(post_poleCells_count[jj].split(', '))
post_poleCells_count_df = pd.DataFrame(post_poleCells_count_int).dropna().transpose()

# Prepare jittered data
jitter = 0.04
ant_y_data = [np.array([i] * len(ant_poleCells_count_df[ant_poleCells_count_df.columns[i]].item())) for i in range(ant_poleCells_count_df.size)]
ant_y_data_jittered = [y + st.t(df=6, scale=jitter).rvs(len(y)) for y in ant_y_data]
ant_y_data_jittered_shifted = [y + st.t(df=6, scale=jitter).rvs(len(y))-0.2 for y in ant_y_data]
post_y_data = [np.array([i] * len(post_poleCells_count_df[post_poleCells_count_df.columns[i]].item())) for i in range(post_poleCells_count_df.size)]
post_y_data_jittered = [y + st.t(df=6, scale=jitter).rvs(len(y)) for y in post_y_data]
post_y_data_jittered_shifted = [y + st.t(df=6, scale=jitter).rvs(len(y))-0.2 for y in post_y_data]

# Generate scatter plots for anterior and posterior pole cell counts
fig, ax = plt.subplots(1, 2, figsize= (5, len(post_poleCells_count_df.columns)*1.2), sharey=True) #initialise plot
for x1, x2, y1, y2, p1, p2 in zip(post_poleCells_count_df, ant_poleCells_count_df, post_y_data_jittered, ant_y_data_jittered):
    ax[0].scatter(post_poleCells_count_df[x1].item(), y1, s = 100, color='white', edgecolors='k', alpha=0.4)
    ax[1].scatter(ant_poleCells_count_df[x2].item(), y2, s = 100, color='k', edgecolors='k', alpha=0.4)
    
    # Mean ± visual span for each condition
    x1_mean = np.mean(post_poleCells_count_df[x1].item())
    x2_mean = np.mean(ant_poleCells_count_df[x2].item())
    ax[0].plot([x1_mean, x1_mean],[np.median(y1-0.15), np.median(y1+0.15)], color = 'black', linewidth = 3, linestyle = ':')
    ax[1].plot([x2_mean, x2_mean],[np.median(y2-0.1), np.median(y2+0.1)], color = 'black', linewidth = 3)
    ax[0].set_xlim(-10,70)
    ax[1].set_xlim(-10,70)

# Set axis ranges and labels
ax[0].yaxis.set_ticklabels(np.concatenate([['0'],ant_poleCells_count_df.columns.values]))
ax[0].tick_params(size=10,width=3)
ax[1].tick_params(size=10,width=3)
ax[0].tick_params(left = False)
ax[1].tick_params(left = False)
axs = plt.gca()
axs.invert_yaxis()

# Save figure
plt.savefig("directory\\"+date+"_poleCellsCounts_subplots.svg", format='svg',dpi=300)
