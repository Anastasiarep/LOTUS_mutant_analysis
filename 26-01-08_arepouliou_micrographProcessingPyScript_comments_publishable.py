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
Claude Haiku 4.5 was used for code organisation
"""

# Import required libraries for data processing, image analysis, and plotting
import numpy as np                                 # numerical operations
import matplotlib.pyplot as plt                    # plotting
import os                                          # file path handling
import skimage                                     # image processing
from skimage import io                             # image I/O
import pandas as pd                                # dataframe operations
import scipy.stats as st                           # statistics
from scipy.stats import pearsonr, norm, binomtest  # specific statistical functions
import scipy.ndimage as ndimage                    # image filtering
from matplotlib.pyplot import cm                   # colormap handling
from matplotlib.cm import get_cmap                 # alternative colormap access
from scipy.interpolate import interp1d             # interpolation for smooth profiles
from matplotlib.font_manager import FontProperties # font customisation for plots
import pickle

# Function to calculate Manders' Coefficients for colocalization
# Inputs:
# - channel_1: 2D array for channel 1 (e.g., HA)
# - channel_2: 2D array for channel 2 (e.g., target)
# - mask: 2D binary mask defining region of interest
# Returns:
# - M1, M2: Manders' coefficients
# - overlap_dice: Dice overlap coefficient
# - overlap_jaccard: Jaccard overlap coefficient
def calculate_coloc_metrics(channel_1, channel_2, mask, threshold_ch1=None, threshold_ch2=None):
    """
    Calculate formal colocalization metrics between two channels.
    
    Parameters:
    - channel_1, channel_2: intensity arrays
    - mask: binary mask for region of interest
    - threshold_ch1, threshold_ch2: optional thresholds (if None, uses 0)
    
    Returns:
    - Dictionary with Manders' M1, M2, and overlap coefficients
    """
    
    # Apply mask
    ch1_masked = channel_1[mask > 0]
    ch2_masked = channel_2[mask > 0]
    
    # Set default thresholds if not provided
    if threshold_ch1 is None:
        threshold_ch1 = 0
    if threshold_ch2 is None:
        threshold_ch2 = 0
    
    # Create binary thresholded images
    ch1_binary = (ch1_masked > threshold_ch1).astype(float)
    ch2_binary = (ch2_masked > threshold_ch2).astype(float)
    
    # Identify colocalized pixels (both channels above threshold)
    coloc_pixels = (ch1_binary * ch2_binary) > 0
    
    # ========================
    # Manders' Coefficients
    # ========================
    # M1: fraction of ch1 intensity in pixels where ch2 is above threshold
    sum_ch1_total = np.sum(ch1_masked)
    sum_ch1_coloc = np.sum(ch1_masked[ch2_binary > 0])
    M1 = sum_ch1_coloc / sum_ch1_total if sum_ch1_total > 0 else 0
    
    # M2: fraction of ch2 intensity in pixels where ch1 is above threshold
    sum_ch2_total = np.sum(ch2_masked)
    sum_ch2_coloc = np.sum(ch2_masked[ch1_binary > 0])
    M2 = sum_ch2_coloc / sum_ch2_total if sum_ch2_total > 0 else 0
    
    # ========================
    # Overlap Coefficients (Binary)
    # ========================
    # Dice coefficient
    intersection = np.sum(coloc_pixels)
    dice = (2 * intersection) / (np.sum(ch1_binary) + np.sum(ch2_binary)) if (np.sum(ch1_binary) + np.sum(ch2_binary)) > 0 else 0
    
    # Jaccard coefficient
    union = np.sum((ch1_binary + ch2_binary) > 0)
    jaccard = intersection / union if union > 0 else 0
    
    # ========================
    # Pearson's r (for comparison)
    # ========================
    # Computed on all pixels (including those below threshold)
    pearson_r = np.corrcoef(ch1_masked, ch2_masked)[0, 1] if len(ch1_masked) > 2 else np.nan
    
    return {
        'M1': M1,                    # Fraction of ch1 colocalizing with ch2
        'M2': M2,                    # Fraction of ch2 colocalizing with ch1
        'Dice': dice,                # Binary overlap
        'Jaccard': jaccard,          # Binary overlap (union-based)
        'Pearson_r': pearson_r,      # For reference
        'n_coloc_pixels': int(intersection),
        'n_ch1_pixels': int(np.sum(ch1_binary)),
        'n_ch2_pixels': int(np.sum(ch2_binary))
    }

# Function to calculate Li's ICQ (Intensity Correlation Quotient)
# No thresholding required - threshold-independent
def calculate_li_icq(channel_1, channel_2, mask):
    """
    Calculate Li's Intensity Correlation Quotient (ICQ).
    Threshold-independent method measuring pixel-by-pixel correlation
    above and below mean intensity.
    
    Returns:
    - ICQ: ranges from -0.5 (segregation) to +0.5 (colocalization)
    - p_value: statistical significance
    """
    
    # Apply mask
    ch1_masked = channel_1[mask > 0]
    ch2_masked = channel_2[mask > 0]
    
    # Calculate mean intensities
    mean_ch1 = np.mean(ch1_masked)
    mean_ch2 = np.mean(ch2_masked)
    
    # Calculate product of differences from mean (PDM)
    pdm = (ch1_masked - mean_ch1) * (ch2_masked - mean_ch2)
    
    # Count pixels with positive PDM (colocalized)
    n_positive = np.sum(pdm > 0)
    n_total = len(pdm)
    
    # Calculate ICQ
    ICQ = (n_positive / n_total) - 0.5
    
    # Estimate p-value using binomial test
    p_value = binomtest(n_positive, n_total, 0.5, alternative='two-sided').pvalue
    
    return {
        'ICQ': ICQ,
        'p_value': p_value,
        'n_positive_pdm': n_positive,
        'n_total_pixels': n_total
    }

"""
Explanation of colocalisation metrics:

Manders' Coefficients (M1, M2):
    measures directional colocalisation aka
    what fraction of one signal is present where the other signal is

    Properties:
        Range: 0 to 1 (no to complete colocalisation)
        Assymetric (M1 v M2) shows is one signal encompasses the other
        Dependent on threshold choice

    Example:
        M1 = 0.85 means 85% of HA signal is where Vasa is present
        M2 = 0.45 means 45% of Vasa signal is where HA is present
    
Dice Coefficient: 
    measures binary overlap aka
    how much do the two thresholded images overlap
    
    Properties:
        Range: 0 to 1 (no to complete colocalisation)
        Symmetric
        Dependent on threshold choice

    Example:
        Dice = 0.64 = (2 × 80) / (100 + 150)
        if HA occupies 100 px, Vasa occupies 150 px, and they overlap in 80
        
Jaccard Coefficient:
    measures overlap ratio (intersection divided by union) aka
    how much of the combined area is shared

    Properties:
        Range: 0 to 1 (no to complete colocalisation)
        Symmetric
        Dependent on threshold choice
        Similar to but stricter than Dice
    
    Example:
        Dice = 0.64 is Jaccard = ~0.47 (for same data)

Pearson's r:
    measures linear correlation between px intensities aka
    whether signals are related at all intensities

    Properties:
        Range: -1 to 0 to 1 (perfect negative to no to perfect positive correlation)
        Symmetric
        Sensitive to noise
        Does not indicate where signals are correlated
    
    Example:
        r = 0.72 means signals are moderately positively correlated,
        but doesn't tell you where they overlap spatially

Li's ICQ (Intensity Correlation Quotient):
    measures px-by-px correlation aka
    do pxs deviate from their means in the same direction
    
    Properties:
        Range: -0.5 to 0 to 0.5 (complete segragation to random distribution to complete colocalisation (always correlated))
        Threshold-independent
        Includes p-value (statistical testing)

Choice: M2 = % Vasa in HA signal
"""

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
        
        # =====================================================
        # Formal colocalization analysis
        # =====================================================
        # M1/M2      0-1           Fraction of ch1/ch2 in colocalized regions
        # Dice       0-1           Binary spatial overlap
        # Jaccard    0-1           Union-based overlap
        # ICQ       -0.5 to +0.5   Threshold-independent correlation
        # Pearson r -1 to +1       Linear correlation

        # When to use which metric:
        # M1/M2      General colocalization
        # Dice       When channels have clear boundaries
        # Jaccard    Conservative overlap estimate
        # ICQ        Statistical significance (includes p-value)
        # Pearson r  Pixel intensity correlation

        # Define intensity thresholds (using same approach as before)
        # ch2 (HA) threshold
        y2_log = np.log10(y2[y2 > 0])
        y2_log_mean, y2_log_std = np.mean(y2_log), np.std(y2_log)
        thres_ch2 = 10**(y2_log_mean + 2*y2_log_std)
        
        # ch3 (target) threshold
        y3_log = np.log10(y3[y3 > 0])
        y3_log_mean, y3_log_std = np.mean(y3_log), np.std(y3_log)
        thres_ch3 = 10**(y3_log_mean + 2*y3_log_std)
        
        # Calculate Manders' coefficients and overlap
        coloc_metrics_23 = calculate_coloc_metrics(Im2_ant, Im3_ant, Mask_ant, 
                                                   threshold_ch1=thres_ch2, 
                                                   threshold_ch2=thres_ch3)
        M1_23 = coloc_metrics_23['M1']
        M2_23 = coloc_metrics_23['M2']
        dice_23 = coloc_metrics_23['Dice']
        jaccard_23 = coloc_metrics_23['Jaccard']
        
        # Li's ICQ (threshold-independent)
        icq_23 = calculate_li_icq(Im2_ant, Im3_ant, Mask_ant)
        ICQ_23 = icq_23['ICQ']
        ICQ_pval_23 = icq_23['p_value']
        
        # Repeat for channel 4 if it exists
        if (Img.shape[2] == 4):
            y4_log = np.log10(y4[y4 > 0])
            y4_log_mean, y4_log_std = np.mean(y4_log), np.std(y4_log)
            thres_ch4 = 10**(y4_log_mean + 2*y4_log_std)
            
            coloc_metrics_24 = calculate_coloc_metrics(Im2_ant, Im4_ant, Mask_ant,
                                                       threshold_ch1=thres_ch2,
                                                       threshold_ch2=thres_ch4)
            M1_24 = coloc_metrics_24['M1']
            M2_24 = coloc_metrics_24['M2']
            dice_24 = coloc_metrics_24['Dice']
            jaccard_24 = coloc_metrics_24['Jaccard']
            
            icq_24 = calculate_li_icq(Im2_ant, Im4_ant, Mask_ant)
            ICQ_24 = icq_24['ICQ']
            ICQ_pval_24 = icq_24['p_value']
        else:
            M1_24 = M2_24 = dice_24 = jaccard_24 = ICQ_24 = ICQ_pval_24 = np.nan

        # Combine results into a single-row DataFrame
        df_row = pd.DataFrame([[ant_int_2, ant_int_3, ant_int_4, corr_23, corr_24,
                                M1_23, M2_23, dice_23, jaccard_23, ICQ_23, ICQ_pval_23,
                                M1_24, M2_24, dice_24, jaccard_24, ICQ_24, ICQ_pval_24]], 
                              index=[file_prefix])
        df_row.columns = ['Ant Int ch 2', 'Ant Int ch 3', 'Ant Int ch 4', 'Pears 2-3', 'Pears 2-4',
                          'M1 2-3', 'M2 2-3', 'Dice 2-3', 'Jaccard 2-3', 'ICQ 2-3', 'ICQ_pval 2-3',
                          'M1 2-4', 'M2 2-4', 'Dice 2-4', 'Jaccard 2-4', 'ICQ 2-4', 'ICQ_pval 2-4']
        return(df_row)    
    
    # Process for calculating anterior enrichment and pairwise correlation
    if experiment == 'signal_distribution':
        # Collapse raw intensities along Y to generate anterior-posterior linescans
        # (average raw intensity per X position, weighted by mask)
        A = np.sum(Mask_cropped*Img_cropped[:,:,0], axis=0)/XPixels_cropped
        B = np.sum(Mask_cropped*Img_cropped[:,:,1], axis=0)/XPixels_cropped
        C = np.sum(Mask_cropped*Img_cropped[:,:,2], axis=0)/XPixels_cropped
        if (Img.shape[2] == 4):
            D = np.sum(Mask_cropped*Img_cropped[:,:,3], axis=0)/XPixels_cropped
        
        # Interpolate to fixed resolution along AP axis (300 points)
        x_ = np.arange(0,len(A))/(len(A)-1)
        X = np.linspace(0,1,300)
        fit_func_A, fit_func_B, fit_func_C = interp1d(x_,A), interp1d(x_,B), interp1d(x_,C)
        if (Img.shape[2] == 4):
            fit_func_D = interp1d(x_,D)
        A_, B_, C_ = fit_func_A(X), fit_func_B(X), fit_func_C(X)
        if (Img.shape[2] == 4):
            D_ = fit_func_D(X)
        
        # Optional: plot raw intensity linescans across AP axis
        if PS == True:
            fig, ax = plt.subplots()
            ax.set_xticks([0, 0.5, 1.0])
            ax.plot(X, B_, 'g.', label='channel_2')
            ax.plot(X, C_, 'b.', label='channel_3')
            if (Img.shape[2] == 4):
                ax.plot(X, D_, 'r.', label='channel_4')
            ax.legend(loc='best')
            ax.set_ylabel('Raw Intensity (a.u.)')
            ax.set_xlabel('Anterior-Posterior Position')
            plt.show()
        
        # Concatenate profile data into a single array (same format as linescan)
        if (Img.shape[2] == 4):
            all_array = np.concatenate((X, A_, B_, C_, D_), axis=None)
        else:
            all_array = np.concatenate((X, A_, B_, C_), axis=None)
        df_row = pd.DataFrame([all_array], index=[file_prefix])
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
            plt.savefig(r"C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\\linescans"+date+"_"+cell+"_"+tissue_stain+"_HA-"+channel+"_ZscoreOverPosition_linescan.png",format='png', dpi=300)
        if (form == "svg"):
            plt.savefig(r'C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\\linescans'+date+"_"+cell+"_"+tissue_stain+"_HA-"+channel+"_ZscoreOverPosition_linescan.svg", format='svg', dpi=300)

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
def return_box(species_plot, labels_plus, coloursPlus, ddf_filtered, cell, stain, date, xlimits, pos_ctrl='84x34', neg_ctrl=np.nan):
    # Extract control data
    if pd.notnull(neg_ctrl):
        TF_array = [neg_ctrl in ind for ind in ddf_filtered.index]
        ddf_neg = np.array(ddf_filtered[TF_array]).ravel()
        med_neg = np.median(ddf_neg)
        pValues_Neg = [float("NaN")]*(len(species_plot))
        pValues_Neg_symbolic = [""]*(len(species_plot))

    # Extract values for positive control
    TF_array = [pos_ctrl in ind for ind in ddf_filtered.index]
    ddf_pos = np.array(ddf_filtered[TF_array]).ravel()
    med_pos = np.median(ddf_pos)
    pValues_Pos = [float("NaN")]*(len(species_plot))
    pValues_Pos_symbolic = [""]*(len(species_plot))

    # Gather data per genotype and perform bootstrap significance tests
    ddf = []
    for k, sp in enumerate(species_plot):
        TF_array = [sp in ind for ind in ddf_filtered.index]
        ddf_specie = np.array(ddf_filtered[TF_array]).ravel()
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
    # Create figure
    textsize = 30
    fig, ax = plt.subplots(figsize=(10, len(species_plot)*1), constrained_layout=True)

    # Plot boxplot with notches
    bp = ax.boxplot(ddf, patch_artist=True, notch=True, vert=False, showfliers=False, widths=0.6)
    ax.invert_yaxis()

    # Style boxes
    for patch, colour in zip(bp['boxes'], coloursPlus):
        patch.set_facecolor(colour)
        patch.set_alpha(0.4)
    
    # Style whiskers, caps, medians (from original)
    for whisker in bp['whiskers']:
        whisker.set(color='grey', linewidth=6, linestyle=':')
    for cap in bp['caps']:
        cap.set(color='grey', linewidth=6)
    for median in bp['medians']:
        median.set(color='k', linewidth=6)
    
    # Overlay jittered points
    jitter = 0.04
    y_data = [np.array([i+1] * len(d)) for i, d in enumerate(ddf)]
    y_data_jittered = [y + st.t(df=6, scale=jitter).rvs(len(y)) for y in y_data]
    for x, y in zip(ddf, y_data_jittered):
        ax.scatter(x, y, marker='.', color='k')
    
    # Set axes
    ax.set_xlim(xlimits)
    xmin, xmax = np.round(ax.get_xlim())
    
    # Add symbolic p-value annotations and mutant labels
    for c in range(len(labels_plus)):
        plt.text(xmax+((xmax-xmin)/10), c+1, pValues_Pos_symbolic[c], fontsize = textsize, color = "black")
        if pd.notnull(neg_ctrl):
            plt.text(xmax, c+1, pValues_Neg_symbolic[c], fontsize = textsize, color="grey")
        # plt.text(xmin-(2*(xmax-xmin)/10), c+1, labels_plus[c], fontsize = textsize, color = "black")
    
    # Remove default y-axis, add manual labels
    # ax.set_yticks([])
    # ax.set_yticklabels([])
    ax.set_yticks(np.arange(1, len(labels_plus) + 1))
    ax.set_yticklabels(labels_plus, fontsize=textsize)
    ax.tick_params(axis='y', left=False, labelleft=True)        

    # X-axis formatting
    ax.set_xticks([xmin, 0, xmax])
    ax.set_xticklabels([int(xmin), 0, int(xmax)], fontsize=textsize)
    ax.tick_params(size=15, width=7)
    
    # Reference lines for controls
    if pd.notnull(neg_ctrl):
        ax.axvline(x=med_neg, color='gray', linewidth=2)
    ax.axvline(x=med_pos, color='black', linewidth=2)
    
    # Title and save
    plt.title(cell + stain, fontsize=textsize)
    plt.savefig(r"C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\\"+date+"_boxplots"+cell+stain+".svg", format='svg',dpi=300)
    plt.show()

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
    bp = ax.boxplot(ddf, boxprops=dict(linewidth=line_width), patch_artist = True, notch =True, vert = False, showfliers=False, widths=0.6)
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
    ax.yaxis.set_ticks([])
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
    plt.savefig(r"C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\\"+date+"_boxplots4fig"+cell+stain+".svg", format='svg',dpi=300)

#............................................................................#
#............................................................................#
#............................................................................#

# Function to extract median raw intensity from full oocyte/embryo
# Inputs:
# - ddf_raw_signals: DataFrame output from 'signal_distribution' experiment
# - channel: integer (1, 2, 3, or 4) specifying which channel to extract
# Returns:
# - Series containing median raw intensity for each sample
def extract_median_raw_intensity(ddf_raw_signals, channel):
    """
    Extract median raw intensity across entire A-P axis for a given channel.
    Channels: 1=ch1, 2=ch2, 3=ch3, 4=ch4
    """
    channel_idx = channel - 1  # Convert to 0-indexed
    start_idx = 300 + (channel_idx * 300)  # Position data ends at 300, then channels start
    end_idx = start_idx + 300
    
    medians = ddf_raw_signals.iloc[:, start_idx:end_idx].median(axis=1)
    return medians

# Function to extract median raw intensity from anterior region only
# Inputs:
# - ddf_raw_signals: DataFrame output from 'signal_distribution' experiment
# - channel: integer specifying which channel
# - anterior_fraction: fraction of A-P axis to define as anterior (default 0.15)
# Returns:
# - Series containing median anterior raw intensity for each sample
def extract_median_raw_intensity_anterior(ddf_raw_signals, channel, anterior_fraction=0.15):
    """
    Extract median raw intensity from anterior region (default: anterior 15% of A-P axis).
    Channels: 1=ch1, 2=ch2, 3=ch3, 4=ch4
    """
    channel_idx = channel - 1
    start_idx = 300 + (channel_idx * 300)
    end_idx = start_idx + 300
    
    # Anterior region: first 15% of interpolated points
    ant_end = int(300 * anterior_fraction)
    
    medians_ant = ddf_raw_signals.iloc[:, start_idx:(start_idx + ant_end)].median(axis=1)
    return medians_ant

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
# sample_dir = "adov_HAVasa\\"
sample_dir = r"D:\LOTUSmut\manually_filtered\adov_HAVasa"

# Identify all unique image filenames (excluding extensions)
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]

# Initialise output DataFrames for this dataset
ddf_oocytes_HAVasa = pd.DataFrame()
ddf_oocytes_HAVasa_linescans = pd.DataFrame()

# Loop over all micrographs and extract both linescan and anterior metrics
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_oocytes_HAVasa_linescans = pd.concat([ddf_oocytes_HAVasa_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_oocytes_HAVasa = pd.concat([ddf_oocytes_HAVasa,df])

output_dir = r"C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\analysis_results"
os.makedirs(output_dir, exist_ok=True)
ddf_oocytes_HAVasa.to_pickle(f'{output_dir}/ddf_oocytes_HAVasa.pkl')
ddf_oocytes_HAVasa.to_csv(f'{output_dir}/ddf_oocytes_HAVasa.csv')
# For subsequent analysis, load saved dataframe
ddf_oocytes_HAVasa = pd.read_pickle(f'{output_dir}\ddf_oocytes_HAVasa.pkl')

# Prepare z-score linescan data for plotting (HA and Vasa channels)
ddf = pd.concat([
    ddf_oocytes_HAVasa_linescans.iloc[:, 0:300],
    ddf_oocytes_HAVasa_linescans.iloc[:, 600:1200]
], axis=1)
ddf.columns = range(0, 900)

# Define plot configuration
date = "26-11-11"
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
sample_dir = r"D:\LOTUSmut\manually_filtered\earlyBros_HAVasa"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HAVasa = pd.DataFrame()
ddf_embryos_HAVasa_linescans = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_HAVasa_linescans = pd.concat([ddf_embryos_HAVasa_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_embryos_HAVasa = pd.concat([ddf_embryos_HAVasa,df])
ddf_embryos_HAVasa.to_pickle(f'{output_dir}/ddf_embryos_HAVasa.pkl')
ddf_embryos_HAVasa.to_csv(f'{output_dir}/ddf_embryos_HAVasa.csv')
# For subsequent analysis, load saved dataframe
ddf_embryos_HAVasa = pd.read_pickle(f'{output_dir}\ddf_embryos_HAVasa.pkl')

# Extract relevant metrics (HA = ch2, Vasa = ch4, Pearson = 2-4)
ddf_embryos_HAVasa_AntInt_HA = ddf_embryos_HAVasa['Ant Int ch 2'].dropna()
ddf_embryos_HAVasa_AntInt_Vasa = ddf_embryos_HAVasa['Ant Int ch 4'].dropna()
ddf_embryos_HAVasa_Pears_HAVasa = ddf_embryos_HAVasa['Pears 2-4'].dropna()
#............................................................................#
# Repeat for HA-nanos embryos
sample_dir = r"D:\LOTUSmut\manually_filtered\earlyBros_HAnanos"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HAnanos = pd.DataFrame()
ddf_embryos_HAnanospgc_linescans = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_HAnanospgc_linescans = pd.concat([ddf_embryos_HAnanospgc_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_embryos_HAnanos = pd.concat([ddf_embryos_HAnanos,df])
ddf_embryos_HAnanos.to_pickle(f'{output_dir}/ddf_embryos_HAnanos.pkl')
ddf_embryos_HAnanos.to_csv(f'{output_dir}/ddf_embryos_HAnanos.csv')
# For subsequent analysis, load saved dataframe
ddf_embryos_HAnanos = pd.read_pickle(f'{output_dir}\ddf_embryos_HAnanos.pkl')

ddf_embryos_HAnanos_AntInt_HA = ddf_embryos_HAnanos['Ant Int ch 2'].dropna()
ddf_embryos_HAnanos_AntInt_nanos = ddf_embryos_HAnanos['Ant Int ch 3'].dropna()
ddf_embryos_HAnanos_Pears_HAnanos = ddf_embryos_HAnanos['Pears 2-3'].dropna()
#............................................................................#
# Repeat for HA-pgc embryos
sample_dir = r"D:\LOTUSmut\manually_filtered\earlyBros_HApgc"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HApgc = pd.DataFrame()
ddf_embryos_HAnanospgc_linescans = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df=return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_HAnanospgc_linescans = pd.concat([ddf_embryos_HAnanospgc_linescans,df])
    df=return_multichannel_spatial_channel_intensity('anterior_metrics', prefix, sample_dir)
    ddf_embryos_HApgc = pd.concat([ddf_embryos_HApgc,df])
ddf_embryos_HApgc.to_pickle(f'{output_dir}/ddf_embryos_HApgc.pkl')
ddf_embryos_HApgc.to_csv(f'{output_dir}/ddf_embryos_HApgc.csv')
# For subsequent analysis, load saved dataframe
ddf_embryos_HApgc = pd.read_pickle(f'{output_dir}\ddf_embryos_HApgc.pkl')

ddf_embryos_HApgc_AntInt_HA = ddf_embryos_HApgc['Ant Int ch 2'].dropna()
ddf_embryos_HApgc_AntInt_pgc = ddf_embryos_HApgc['Ant Int ch 4'].dropna()
ddf_embryos_HApgc_Pears_HApgc = ddf_embryos_HApgc['Pears 2-4'].dropna()

# --------------------------------------------------------------------------
# Section 2b: Process raw signal data for all datasets
# --------------------------------------------------------------------------

# Vasa (oocytes)
sample_dir = r"D:\LOTUSmut\manually_filtered\adov_HAVasa"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_oocytes_HAVasa_raw_signals = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df = return_multichannel_spatial_channel_intensity('signal_distribution', prefix, sample_dir)
    ddf_oocytes_HAVasa_raw_signals = pd.concat([ddf_oocytes_HAVasa_raw_signals, df])
ddf_oocytes_HAVasa_raw_signals.to_pickle(f'{output_dir}/ddf_oocytes_HAVasa_raw_signals.pkl')
ddf_oocytes_HAVasa_raw_signals.to_csv(f'{output_dir}/ddf_oocytes_HAVasa_raw_signals.csv')
# For subsequent analysis, load saved dataframe
ddf_oocytes_HAVasa_raw_signals = pd.read_pickle(f'{output_dir}\ddf_oocytes_HAVasa_raw_signals.pkl')

# Vasa (embryos)
sample_dir = r"D:\LOTUSmut\manually_filtered\earlyBros_HAVasa"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HAVasa_raw_signals = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df = return_multichannel_spatial_channel_intensity('signal_distribution', prefix, sample_dir)
    ddf_embryos_HAVasa_raw_signals = pd.concat([ddf_embryos_HAVasa_raw_signals, df])
ddf_embryos_HAVasa_raw_signals.to_pickle(f'{output_dir}/ddf_embryos_HAVasa_raw_signals.pkl')
ddf_embryos_HAVasa_raw_signals.to_csv(f'{output_dir}/ddf_embryos_HAVasa_raw_signals.csv')
# For subsequent analysis, load saved dataframe
ddf_embryos_HAVasa_raw_signals = pd.read_pickle(f'{output_dir}\ddf_embryos_HAVasa_raw_signals.pkl')

# nanos (embryos)
sample_dir = r"D:\LOTUSmut\manually_filtered\earlyBros_HAnanos"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HAnanos_raw_signals = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df = return_multichannel_spatial_channel_intensity('signal_distribution', prefix, sample_dir)
    ddf_embryos_HAnanos_raw_signals = pd.concat([ddf_embryos_HAnanos_raw_signals, df])
ddf_embryos_HAnanos_raw_signals.to_pickle(f'{output_dir}/ddf_embryos_HAnanos_raw_signals.pkl')
ddf_embryos_HAnanos_raw_signals.to_csv(f'{output_dir}/ddf_embryos_HAnanos_raw_signals.csv')
# For subsequent analysis, load saved dataframe
ddf_embryos_HAnanos_raw_signals = pd.read_pickle(f'{output_dir}\ddf_embryos_HAnanos_raw_signals.pkl')

# pgc (embryos)
sample_dir = r"D:\LOTUSmut\manually_filtered\earlyBros_HApgc"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_embryos_HApgc_raw_signals = pd.DataFrame()
for i, prefix in enumerate(UnqFiles_prefixes, 1):
    if i % 50 == 0:
        print(f"{i}/{len(UnqFiles_prefixes)} files processed")
    # print(prefix)
    df = return_multichannel_spatial_channel_intensity('signal_distribution', prefix, sample_dir)
    ddf_embryos_HApgc_raw_signals = pd.concat([ddf_embryos_HApgc_raw_signals, df])
ddf_embryos_HApgc_raw_signals.to_pickle(f'{output_dir}/ddf_embryos_HApgc_raw_signals.pkl')
ddf_embryos_HApgc_raw_signals.to_csv(f'{output_dir}/ddf_embryos_HApgc_raw_signals.csv')
# For subsequent analysis, load saved dataframe
ddf_embryos_HApgc_raw_signals = pd.read_pickle(f'{output_dir}\ddf_embryos_HApgc_raw_signals.pkl')

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
date = "26-11-11"
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
date = "26-11-11"
channel = 'nanos'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "all_embryos"
tissue_stain = 'nanos_extra'
ymin=-2
ymax=15
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg", extra='84x34_m22-12-25_sl2_ser34')
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg")

#............................................................................#
# pgc (embryos)
ddf = pd.concat([
    ddf_embryos_HAnanospgc_linescans.iloc[:, 0:300],
    ddf_embryos_HAnanospgc_linescans.iloc[:, 600:900],
    ddf_embryos_HAnanospgc_linescans.iloc[:, 1200:1500]
], axis=1)
ddf.columns = range(0, 900)
date = "26-11-11"
channel = 'pgc'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "all_embryos"
tissue_stain = 'pgc_extra'
ymin=-2
ymax=15
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg", extra='84x34_m22-12-25_sl2_ser34')
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg")

# --------------------------------------------------------------------------
# Section 3b: Generate raw signal linescan plots for HA only (all experiments)
# --------------------------------------------------------------------------

def plot_raw_intensity_linescans_single(species_plot, labels, colours, channel, ddf, date, cell, tissue_stain, PS=False, form="svg"):
    """
    Plot raw intensity linescans (single channel per genotype).
    Auto-scales y-axis for each subplot.
    
    Parameters:
    - species_plot: list of genotype identifiers
    - labels: display labels for each genotype
    - colours: colours for each genotype
    - channel: channel name for title
    - ddf: DataFrame with data (columns 0-299=X, 300-599=intensity)
    - date, cell, tissue_stain: for file naming
    - PS: save figure if True
    - form: file format ('svg' or 'png')
    """
    
    n_conditions = len(species_plot)
    font_size = 6
    tick_width = 0.5
    tick_size = 2
    line_width = 1
    
    fig, ax = plt.subplots(nrows=n_conditions, ncols=1, figsize=(3, n_conditions*0.6), sharex='col')
    
    for specie_i in range(n_conditions):
        # Get data for this genotype
        specie_plottables = return_mean_cis(ddf, species_plot[specie_i])
        
        X_pos = specie_plottables[0][0:300]
        intensity_mean = specie_plottables[0][300:600]
        intensity_lower = specie_plottables[1][300:600]
        intensity_upper = specie_plottables[2][300:600]
        
        n_samples = int(specie_plottables[4][0])
        
        # Plot
        ax[specie_i].plot(X_pos, intensity_mean, color=colours[specie_i], linewidth=line_width, alpha=1)
        ax[specie_i].fill_between(list(X_pos), intensity_lower, intensity_upper, color=colours[specie_i], alpha=0.2)
        
        # Auto-scale y-axis with 5% padding
        ymin = intensity_lower.min()
        ymax = intensity_upper.max()
        ypadding = (ymax - ymin) * 0.05
        ax[specie_i].set_ylim(ymin - ypadding, ymax + ypadding)
        
        # Format
        # ax[specie_i].tick_params(size=tick_size, width=tick_width)
        # ax[specie_i].grid(True, alpha=0.2, linestyle='--')
        # ax[specie_i].set_ylabel(labels[specie_i], fontsize=font_size, fontweight='bold')
        # Format: minimal
        # ax[specie_i].tick_params(size=tick_size, width=tick_width, labelbottom=False, labelleft=True, labelsize=font_size)
        ax[specie_i].tick_params(size=tick_size, width=tick_width, labelbottom=False, labelleft=False, labelsize=font_size)
        ax[specie_i].spines['top'].set_visible(False)
        ax[specie_i].spines['right'].set_visible(False)
        
        # Sample size annotation
        # ax[specie_i].text(0.02, 0.95, f'N={n_samples}', transform=ax[specie_i].transAxes, 
        #                  fontsize=font_size-2, verticalalignment='top', 
        #                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Format x-axis
    # ax[-1].set_xticks([0, 0.5, 1.0])
    # ax[-1].set_xlabel('Anterior-Posterior Position', fontsize=font_size)
    # ax[-1].tick_params(labelbottom=True)
    # ax[-1].set_xticks([0, 0.5, 1.0])
    
    plt.suptitle(f'{tissue_stain} - {channel}', fontsize=font_size+2, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if (PS == True):
        if (form == "png"):
            plt.savefig(r"C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\\linescans"+date+"_"+cell+"_"+tissue_stain+"_HA-"+channel+"_OverPosition_autoscale_linescan.png",format='png', dpi=300)
        if (form == "svg"):
            plt.savefig(r'C:\Users\anrep\OneDrive\Documents\Harvard\extavour_LOTUS_MS\260108_plots-figures\\linescans'+date+"_"+cell+"_"+tissue_stain+"_HA-"+channel+"_OverPosition_autoscle_linescan.svg", format='svg', dpi=300)

    plt.show()

colours = ['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
species_plot = ['84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']

# Vasa (oocytes) - HA channel only
ddf_raw = pd.concat([
    ddf_oocytes_HAVasa_raw_signals.iloc[:, 0:300],      # X positions
    ddf_oocytes_HAVasa_raw_signals.iloc[:, 600:900]     # HA intensity
], axis=1)
ddf_raw.columns = range(0, 600)

# Plot
plot_raw_intensity_linescans_single(
    species_plot, labels, colours, 'Vasa', ddf_raw, 
    "26-01-13", "all_adov", 'Vasa_HA_raw', PS=True, form="svg"
)

# Vasa (embryos) - HA channel only
ddf_raw = pd.concat([
    ddf_embryos_HAVasa_raw_signals.iloc[:, 0:300],      # X positions
    ddf_embryos_HAVasa_raw_signals.iloc[:, 600:900]     # HA intensity
], axis=1)
ddf_raw.columns = range(0, 600)
plot_raw_intensity_linescans_single(
    species_plot, labels, colours, 'Vasa', ddf_raw, 
    "26-01-13", "all_embryos", 'Vasa_HA_raw', PS=True, form="svg"
)

# nanos (embryos) - HA channel only
ddf_raw = pd.concat([
    ddf_embryos_HAnanos_raw_signals.iloc[:, 0:300],      # X positions
    ddf_embryos_HAnanos_raw_signals.iloc[:, 600:900]     # HA intensity
], axis=1)
ddf_raw.columns = range(0, 600)
plot_raw_intensity_linescans_single(
    species_plot, labels, colours, 'nanos', ddf_raw, 
    "26-01-13", "all_embryos", 'nanos_HA_raw', PS=True, form="svg"
)

# pgc (embryos) - HA channel only
ddf_raw = pd.concat([
    ddf_embryos_HApgc_raw_signals.iloc[:, 0:300],      # X positions
    ddf_embryos_HApgc_raw_signals.iloc[:, 600:900]     # HA intensity
], axis=1)
ddf_raw.columns = range(0, 600)
plot_raw_intensity_linescans_single(
    species_plot, labels, colours, 'pgc', ddf_raw, 
    "26-01-13", "all_embryos", 'pgc_HA_raw', PS=True, form="svg"
)

# --------------------------------------------------------------------------
# Section 4: Plot integrated anterior enrichment values
# --------------------------------------------------------------------------

date_today = '26-11-11'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['gray']+['white']*14+['k']
cond_cell = "oocyte"
Vasa_adov_xlimits = [-75000, 200000]
stain_cell = 'Vasa_ant_int'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_AntInt_Vasa,
                    'oocyte', 'Vasa_ant_int', date_today, Vasa_adov_xlimits, 
                    pos_ctrl='84x34', neg_ctrl='OR')
return_box_figure(species_plot, labels, colours, ddf_oocytes_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_adov_xlimits, neg_ctrl = 'OR')

#............................................................................#
cond_cell = "bros"
stain_cell = 'Vasa_ant_int'
Vasa_bros_xlimits = [-265000, 295000]
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_bros_xlimits, neg_ctrl = 'OR')
return_box_figure(species_plot, labels, colours, ddf_embryos_HAVasa_AntInt_Vasa, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=Vasa_bros_xlimits, neg_ctrl = 'OR')
#............................................................................#
stain_cell = 'nanos_ant_int'
nanos_bros_xlimits = [-140000, 220000]
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_AntInt_nanos, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=nanos_bros_xlimits, neg_ctrl = 'OR')
return_box_figure(species_plot, labels, colours, ddf_embryos_HAnanos_AntInt_nanos, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=nanos_bros_xlimits, neg_ctrl = 'OR')
#............................................................................#
stain_cell = 'pgc_ant_int'
pgc_bros_xlimits = [-149000, 165000]
return_box(species_plot, labels, colours, ddf_embryos_HApgc_AntInt_pgc, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=pgc_bros_xlimits, neg_ctrl = 'OR')
return_box_figure(species_plot, labels, colours, ddf_embryos_HApgc_AntInt_pgc, cell=cond_cell, stain=stain_cell,date=date_today, xlimits=pgc_bros_xlimits, neg_ctrl = 'OR')

# --------------------------------------------------------------------------
# Section 4b: Extract and plot raw HA signal distributions (boxplots)
# --------------------------------------------------------------------------

date_today = '26-11-11'
species_plot = ['84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['white']*14+['k']

# ============================================================================
# OOCYTES - Vasa - HA raw signal (whole tissue median)
# ============================================================================
ddf_oocytes_HAVasa_raw_median = extract_median_raw_intensity(ddf_oocytes_HAVasa_raw_signals, channel=2)
cond_cell = "oocyte"
stain_cell = 'Vasa_HA_raw_median'
HA_oocytes_xlimits = [round(np.min(ddf_oocytes_HAVasa_raw_median) / 1000) * 1000, 
                      round(np.max(ddf_oocytes_HAVasa_raw_median) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_raw_median, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_oocytes_xlimits)
#............................................................................#
# OOCYTES - Vasa - HA raw signal (anterior median)
# ============================================================================
ddf_oocytes_HAVasa_raw_median_ant = extract_median_raw_intensity_anterior(ddf_oocytes_HAVasa_raw_signals, channel=2, anterior_fraction=0.15)
stain_cell = 'Vasa_HA_raw_median_ant'
HA_oocytes_xlimits = [round(np.min(ddf_oocytes_HAVasa_raw_median_ant) / 1000) * 1000, 
                      round(np.max(ddf_oocytes_HAVasa_raw_median_ant) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_raw_median_ant, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_oocytes_xlimits)

# ============================================================================
# EMBRYOS - Vasa - HA raw signal (whole tissue median)
# ============================================================================
ddf_embryos_HAVasa_raw_median = extract_median_raw_intensity(ddf_embryos_HAVasa_raw_signals, channel=2)
cond_cell = "embryo"
stain_cell = 'Vasa_HA_raw_median'
HA_embryos_xlimits = [round(np.min(ddf_embryos_HAVasa_raw_median) / 1000) * 1000, 
                      round(np.max(ddf_embryos_HAVasa_raw_median) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_raw_median, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_embryos_xlimits)
#............................................................................#
# EMBRYOS - Vasa - HA raw signal (anterior median)
# ============================================================================
ddf_embryos_HAVasa_raw_median_ant = extract_median_raw_intensity_anterior(ddf_embryos_HAVasa_raw_signals, channel=2, anterior_fraction=0.15)
stain_cell = 'Vasa_HA_raw_median_ant'
HA_embryos_xlimits = [round(np.min(ddf_embryos_HAVasa_raw_median_ant) / 1000) * 1000, 
                      round(np.max(ddf_embryos_HAVasa_raw_median_ant) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_raw_median_ant, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_embryos_xlimits)

# ============================================================================
# EMBRYOS - nanos - HA raw signal (whole tissue median)
# ============================================================================
ddf_embryos_HAnanos_raw_median = extract_median_raw_intensity(ddf_embryos_HAnanos_raw_signals, channel=2)
cond_cell = "embryo"
stain_cell = 'nanos_HA_raw_median'
HA_nanos_xlimits = [round(np.min(ddf_embryos_HAnanos_raw_median) / 1000) * 1000, 
                      round(np.max(ddf_embryos_HAnanos_raw_median) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_raw_median, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_nanos_xlimits)
#............................................................................#
# EMBRYOS - nanos - HA raw signal (anterior median)
# ============================================================================
ddf_embryos_HAnanos_raw_median_ant = extract_median_raw_intensity_anterior(ddf_embryos_HAnanos_raw_signals, channel=2, anterior_fraction=0.15)
stain_cell = 'nanos_HA_raw_median_ant'
HA_nanos_xlimits = [round(np.min(ddf_embryos_HAnanos_raw_median_ant) / 1000) * 1000, 
                      round(np.max(ddf_embryos_HAnanos_raw_median_ant) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_raw_median_ant, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_nanos_xlimits)

# ============================================================================
# EMBRYOS - pgc - HA raw signal (whole tissue median)
# ============================================================================
ddf_embryos_HApgc_raw_median = extract_median_raw_intensity(ddf_embryos_HApgc_raw_signals, channel=2)
cond_cell = "embryo"
stain_cell = 'pgc_HA_raw_median'
HA_pgc_xlimits = [round(np.min(ddf_embryos_HApgc_raw_median) / 1000) * 1000, 
                      round(np.max(ddf_embryos_HApgc_raw_median) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_embryos_HApgc_raw_median, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_pgc_xlimits)
#............................................................................#
# EMBRYOS - pgc - HA raw signal (anterior median)
# ============================================================================
ddf_embryos_HApgc_raw_median_ant = extract_median_raw_intensity_anterior(ddf_embryos_HApgc_raw_signals, channel=2, anterior_fraction=0.15)
stain_cell = 'pgc_HA_raw_median_ant'
HA_pgc_xlimits = [round(np.min(ddf_embryos_HApgc_raw_median_ant) / 1000) * 1000, 
                      round(np.max(ddf_embryos_HApgc_raw_median_ant) / 1000) * 1000]
return_box(species_plot, labels, colours, ddf_embryos_HApgc_raw_median_ant, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_pgc_xlimits)

# --------------------------------------------------------------------------
# Section 4c: Plot normalized HA signal distributions (boxplots)
# Normalized to negative control (OR) median
# --------------------------------------------------------------------------

date_today = '26-11-11'
species_plot = ['84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['white']*14+['k']

# ============================================================================
# OOCYTES - Vasa - HA raw signal normalized (whole tissue median)
# ============================================================================
# Extract OR median and normalize all values to it
OR_median_oocytes_vasa = ddf_oocytes_HAVasa_raw_median[ddf_oocytes_HAVasa_raw_median.index.str.contains('OR')].median()
ddf_oocytes_HAVasa_raw_median_norm = ddf_oocytes_HAVasa_raw_median / OR_median_oocytes_vasa

cond_cell = "oocyte"
stain_cell = 'Vasa_HA_raw_median_norm'
HA_oocytes_norm_xlimits = [round(np.min(ddf_oocytes_HAVasa_raw_median_norm)), 
                           round(np.max(ddf_oocytes_HAVasa_raw_median_norm))]
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_raw_median_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_oocytes_norm_xlimits)
# OOCYTES - Vasa - HA raw signal normalized (anterior median)
# ============================================================================
OR_median_oocytes_vasa_ant = ddf_oocytes_HAVasa_raw_median_ant[ddf_oocytes_HAVasa_raw_median_ant.index.str.contains('OR')].median()
ddf_oocytes_HAVasa_raw_median_ant_norm = ddf_oocytes_HAVasa_raw_median_ant / OR_median_oocytes_vasa_ant

stain_cell = 'Vasa_HA_raw_median_ant_norm'
HA_oocytes_norm_xlimits = [round(np.min(ddf_oocytes_HAVasa_raw_median_ant_norm)), round(np.max(ddf_oocytes_HAVasa_raw_median_ant_norm))]
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa_raw_median_ant_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_oocytes_norm_xlimits)

# ============================================================================
# EMBRYOS - Vasa - HA raw signal normalized (whole tissue median)
# ============================================================================
OR_median_embryos_vasa = ddf_embryos_HAVasa_raw_median[ddf_embryos_HAVasa_raw_median.index.str.contains('OR')].median()
ddf_embryos_HAVasa_raw_median_norm = ddf_embryos_HAVasa_raw_median / OR_median_embryos_vasa

cond_cell = "embryo"
stain_cell = 'Vasa_HA_raw_median_norm'
HA_embryos_norm_xlimits = [round(np.min(ddf_embryos_HAVasa_raw_median_norm)), round(np.max(ddf_embryos_HAVasa_raw_median_norm))]
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_raw_median_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_embryos_norm_xlimits)
# EMBRYOS - Vasa - HA raw signal normalized (anterior median)
# ============================================================================
OR_median_embryos_vasa_ant = ddf_embryos_HAVasa_raw_median_ant[ddf_embryos_HAVasa_raw_median_ant.index.str.contains('OR')].median()
ddf_embryos_HAVasa_raw_median_ant_norm = ddf_embryos_HAVasa_raw_median_ant / OR_median_embryos_vasa_ant

stain_cell = 'Vasa_HA_raw_median_ant_norm'
HA_embryos_norm_xlimits = [round(np.min(ddf_embryos_HAVasa_raw_median_ant_norm)), round(np.max(ddf_embryos_HAVasa_raw_median_ant_norm))]
return_box(species_plot, labels, colours, ddf_embryos_HAVasa_raw_median_ant_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_embryos_norm_xlimits)

# ============================================================================
# EMBRYOS - nanos - HA raw signal normalized (whole tissue median)
# ============================================================================
OR_median_embryos_nanos = ddf_embryos_HAnanos_raw_median[ddf_embryos_HAnanos_raw_median.index.str.contains('OR')].median()
ddf_embryos_HAnanos_raw_median_norm = ddf_embryos_HAnanos_raw_median / OR_median_embryos_nanos

cond_cell = "embryo"
stain_cell = 'nanos_HA_raw_median_norm'
HA_nanos_norm_xlimits = [round(np.min(ddf_embryos_HAnanos_raw_median_norm)), round(np.max(ddf_embryos_HAnanos_raw_median_norm))]
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_raw_median_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_nanos_norm_xlimits)
# EMBRYOS - nanos - HA raw signal normalized (anterior median)
# ============================================================================
OR_median_embryos_nanos_ant = ddf_embryos_HAnanos_raw_median_ant[ddf_embryos_HAnanos_raw_median_ant.index.str.contains('OR')].median()
ddf_embryos_HAnanos_raw_median_ant_norm = ddf_embryos_HAnanos_raw_median_ant / OR_median_embryos_nanos_ant

stain_cell = 'nanos_HA_raw_median_ant_norm'
HA_nanos_norm_xlimits = [round(np.min(ddf_embryos_HAnanos_raw_median_ant_norm)), round(np.max(ddf_embryos_HAnanos_raw_median_ant_norm))]
return_box(species_plot, labels, colours, ddf_embryos_HAnanos_raw_median_ant_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_nanos_norm_xlimits)

# ============================================================================
# EMBRYOS - pgc - HA raw signal normalized (whole tissue median)
# ============================================================================
OR_median_embryos_pgc = ddf_embryos_HApgc_raw_median[ddf_embryos_HApgc_raw_median.index.str.contains('OR')].median()
ddf_embryos_HApgc_raw_median_norm = ddf_embryos_HApgc_raw_median / OR_median_embryos_pgc

cond_cell = "embryo"
stain_cell = 'pgc_HA_raw_median_norm'
HA_pgc_norm_xlimits = [round(np.min(ddf_embryos_HApgc_raw_median_norm)), round(np.max(ddf_embryos_HApgc_raw_median_norm))]
return_box(species_plot, labels, colours, ddf_embryos_HApgc_raw_median_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_pgc_norm_xlimits)
# EMBRYOS - pgc - HA raw signal normalized (anterior median)
# ============================================================================
OR_median_embryos_pgc_ant = ddf_embryos_HApgc_raw_median_ant[ddf_embryos_HApgc_raw_median_ant.index.str.contains('OR')].median()
ddf_embryos_HApgc_raw_median_ant_norm = ddf_embryos_HApgc_raw_median_ant / OR_median_embryos_pgc_ant

stain_cell = 'pgc_HA_raw_median_ant_norm'
HA_pgc_norm_xlimits = [round(np.min(ddf_embryos_HApgc_raw_median_ant_norm)), round(np.max(ddf_embryos_HApgc_raw_median_ant_norm))]
return_box(species_plot, labels, colours, ddf_embryos_HApgc_raw_median_ant_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_pgc_norm_xlimits)

# --------------------------------------------------------------------------
# Section 5: Plot Pearson’s correlation coefficients
# --------------------------------------------------------------------------

date_today = '26-11-11'

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

# --------------------------------------------------------------------------
# Section 5b: Plot formal colocalization metrics
# --------------------------------------------------------------------------

date_today = '26-11-11'
species_plot = ['84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['white']*14+['k']

# Oocyte Vasa - Manders M1
cond_cell = "oocyte"
stain_cell = 'Vasa_M1_2-3'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa['M1 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Oocyte Vasa - Manders M2
stain_cell = 'Vasa_M2_2-3'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa['M2 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Oocyte Vasa - Dice overlap
stain_cell = 'Vasa_Dice_2-3'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa['Dice 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Oocyte Vasa - ICQ (Li's)
stain_cell = 'Vasa_ICQ_2-3'
return_box(species_plot, labels, colours, ddf_oocytes_HAVasa['ICQ 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[-0.5, 1])

# Embryo Vasa - Manders M1
cond_cell = "embryo"
stain_cell = 'Vasa_M1_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAVasa['M1 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo Vasa - Manders M2
stain_cell = 'Vasa_M2_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAVasa['M2 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo Vasa - Dice overlap
stain_cell = 'Vasa_Dice_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAVasa['Dice 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo Vasa - ICQ (Li's)
stain_cell = 'Vasa_ICQ_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAVasa['ICQ 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[-0.5, 1])

# Embryo nanos - Manders M1
cond_cell = "embryo"
stain_cell = 'nanos_M1_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAnanos['M1 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo nanos - Manders M2
stain_cell = 'nanos_M2_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAnanos['M2 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo nanos - Dice overlap
stain_cell = 'nanos_Dice_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAnanos['Dice 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo nanos - ICQ (Li's)
stain_cell = 'nanos_ICQ_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HAnanos['ICQ 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[-0.5, 1])

# Embryo pgc - Manders M1
cond_cell = "embryo"
stain_cell = 'pgc_M1_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HApgc['M1 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo pgc - Manders M2
stain_cell = 'pgc_M2_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HApgc['M2 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo pgc - Dice overlap
stain_cell = 'pgc_Dice_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HApgc['Dice 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[0, 1])
# Embryo pgc - ICQ (Li's)
stain_cell = 'pgc_ICQ_2-3'
return_box(species_plot, labels, colours, ddf_embryos_HApgc['ICQ 2-3'].dropna(), cell=cond_cell, stain=stain_cell, date=date_today, xlimits=[-0.5, 0.5])
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

# --------------------------------------------------------------------------
# Section 5: HA signal distribution in pole cell stage embryos
# --------------------------------------------------------------------------

# Process raw signal data for pole cell stage embryos
sample_dir = "lateBros_HA\\"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
ddf_polecells_HA_raw_signals = pd.DataFrame()
for prefix in UnqFiles_prefixes:
    print(prefix)
    df = return_multichannel_spatial_channel_intensity('signal_distribution', prefix, sample_dir)
    ddf_polecells_HA_raw_signals = pd.concat([ddf_polecells_HA_raw_signals, df])

# Extract median raw HA signal values (whole tissue and anterior)
ddf_polecells_HA_raw_median = extract_median_raw_intensity(ddf_polecells_HA_raw_signals, channel=2)
ddf_polecells_HA_raw_median_ant = extract_median_raw_intensity_anterior(ddf_polecells_HA_raw_signals, channel=2, anterior_fraction=0.15)

#............................................................................#
# Plot raw HA signal boxplots - pole cell stage embryos
#............................................................................#

date_today = '26-11-11'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x38', '84x39', '84x40', '84x41', '84x42', '84x43', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'Triple', 'WT Osk']
colours = ['gray']+['white']*14+['k']

# ============================================================================
# POLE CELLS - HA raw signal (whole tissue median)
# ============================================================================
cond_cell = "polecells"
stain_cell = 'HA_raw_median'
HA_polecells_xlimits = [0, 600]  # Adjust based on your data range
return_box(species_plot, labels, colours, ddf_polecells_HA_raw_median, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_polecells_xlimits, neg_ctrl='OR')

#............................................................................#
# POLE CELLS - HA raw signal (anterior median)
# ============================================================================
stain_cell = 'HA_raw_median_ant'
return_box(species_plot, labels, colours, ddf_polecells_HA_raw_median_ant, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_polecells_xlimits, neg_ctrl='OR')

#............................................................................#
# Plot normalized HA signal boxplots - pole cell stage embryos
#............................................................................#

# ============================================================================
# POLE CELLS - HA raw signal normalized (whole tissue median)
# ============================================================================
OR_median_polecells = ddf_polecells_HA_raw_median[ddf_polecells_HA_raw_median.index.str.contains('OR')].median()
ddf_polecells_HA_raw_median_norm = ddf_polecells_HA_raw_median / OR_median_polecells

stain_cell = 'HA_raw_median_norm'
HA_polecells_norm_xlimits = [0, 3]  # Adjust based on your fold-change range
return_box(species_plot, labels, colours, ddf_polecells_HA_raw_median_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_polecells_norm_xlimits, neg_ctrl='OR')

#............................................................................#
# POLE CELLS - HA raw signal normalized (anterior median)
# ============================================================================
OR_median_polecells_ant = ddf_polecells_HA_raw_median_ant[ddf_polecells_HA_raw_median_ant.index.str.contains('OR')].median()
ddf_polecells_HA_raw_median_ant_norm = ddf_polecells_HA_raw_median_ant / OR_median_polecells_ant

stain_cell = 'HA_raw_median_ant_norm'
return_box(species_plot, labels, colours, ddf_polecells_HA_raw_median_ant_norm, cell=cond_cell, stain=stain_cell, date=date_today, xlimits=HA_polecells_norm_xlimits, neg_ctrl='OR')

#............................................................................#
# Generate linescan plots for raw HA signal - pole cell stage embryos
#............................................................................#

ddf_raw = pd.concat([
    ddf_polecells_HA_raw_signals.iloc[:, 0:300],      # X positions
    ddf_polecells_HA_raw_signals.iloc[:, 300:600]     # HA channel (ch2)
], axis=1)
ddf_raw.columns = range(0, 600)

date = "26-11-11"
channel = 'PoleCells'
species_plot = ['OR', '84x133', '84x134', '84x135', '84x136', '84x137', '84x35', '84x36', '84x37', '84x43', '84x38', '84x39', '84x40', '84x41', '84x42', '84x34']
labels = ['OR', 'ΔLOTUS', 'ΔOSK', 'Gbosk', 'VcHlyU', 'DmLis', 'D155A', 'S158R', 'L165A', 'Triple', 'T196S', 'D197N', 'S210P', 'R215Q', 'R215E', 'WT Osk']
colours = ['gray']+['#B3A735']*5+['#089099']*3+['#7CCBA2']*6+['k']
cell = "polecells"
tissue_stain = 'HA_raw'
ymin = 0
ymax = 500  # Adjust based on your data range
return_plotted_ZscoreLineScans(species_plot, labels, colours, channel, ddf_raw, date, cell, tissue_stain, ymin, ymax, PS=True, form="svg")


# -----------------------------------------------------------------------------
# ANALYSIS: Oskar-GFP v HA-Oskar spatial profile
# Single genotype: Oskar-GFP transgenic with WT Oskar-HA-bcd3UTR
# Measures HA and Oskar-GFP distribution across anterior-posterior axis
# -----------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Section 1: Extract Oskar-GFP and HA linescans
# --------------------------------------------------------------------------

sample_dir = "C:\\Users\\anr612\\Downloads\\LOTUS-MS\\GFPOskar\\"
tmp = os.listdir(sample_dir + '\\Rotated')
Files = [ele for ele in tmp if '.tif' in ele]
UnqFiles = list(set(Files))
UnqFiles_prefixes = [name.removeprefix("rotated_").removesuffix(".tif") for name in UnqFiles]
# Initialise output DataFrames for this dataset
ddf_embryos_oskGFPHA_raw = pd.DataFrame()
ddf_embryos_oskGFPHA_linescans = pd.DataFrame()

# Loop over all micrographs and extract both linescan and anterior metrics
for prefix in UnqFiles_prefixes:
    print(prefix)
    df = return_multichannel_spatial_channel_intensity('signal_distribution', prefix, sample_dir)
    ddf_embryos_oskGFPHA_raw = pd.concat([ddf_embryos_oskGFPHA_raw, df])
    df = return_multichannel_spatial_channel_intensity('linescan', prefix, sample_dir)
    ddf_embryos_oskGFPHA_linescans = pd.concat([ddf_embryos_oskGFPHA_linescans, df])

# Prepare z-score linescan data for plotting (Oskar-GFP and HA channels)
# Prepare data for plotting
# For z-scores: X positions (0:300), Oskar-GFP (ch2: 600:900), HA (ch3: 900:1200)
ddf = pd.concat([
    ddf_embryos_oskGFPHA_linescans.iloc[:, 0:300],
    ddf_embryos_oskGFPHA_linescans.iloc[:, 600:1200]
], axis=1)
ddf.columns = range(0, 900)

#............................................................................#
# Plot: Z-score normalized linescans (HA and Oskar-GFP)
#............................................................................#

date = "26-11-11"
channel = 'Oskar-GFP'
cell = "OskarGFP"
tissue_stain = 'OskarGFP-HA'
ymin = -2
ymax = 8

font_size = 6
tick_width = 0.5
tick_size = 2
line_width = 0.5
font = FontProperties()
font.set_name('Arial')

# Extract mean and CI
specie_plottables = return_mean_cis(ddf, 'oskGFP')

X_pos = specie_plottables[0][0:300]
OskarGFP_mean = specie_plottables[0][300:600]
OskarGFP_lower = specie_plottables[1][300:600]
OskarGFP_upper = specie_plottables[2][300:600]

HA_mean = specie_plottables[0][600:900]
HA_lower = specie_plottables[1][600:900]
HA_upper = specie_plottables[2][600:900]

n_samples = int(specie_plottables[4][0])

# Initialise figure with two subplots (one row, two columns)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(2*4, 3), sharex='col', sharey='row')

# Plot Oskar-GFP (left panel)
ax[0].plot(X_pos, OskarGFP_mean, color='k', linewidth=line_width, alpha=1, label='Oskar-GFP')
ax[0].fill_between(list(X_pos), OskarGFP_lower, OskarGFP_upper, color='gray', alpha=0.2)
ax[0].set_ylim(ymin, ymax)
ax[0].set_title('Oskar-GFP', fontsize=font_size+2, fontweight='bold')
ax[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

# Plot HA (right panel)
ax[1].plot(X_pos, HA_mean, color='k', linewidth=line_width, alpha=1, label='HA')
ax[1].fill_between(list(X_pos), HA_lower, HA_upper, color='gray', alpha=0.2)
ax[1].set_ylim(ymin, ymax)
ax[1].set_title('HA', fontsize=font_size+2, fontweight='bold')
ax[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

# Format both axes
for i in range(2):
    ax[i].set_ylim(ymin, ymax)
    ax[i].yaxis.set_ticks([int(ymin), int((ymax+ymin)/2), int(ymax)])
    # ax[i].yaxis.set_ticklabels(['', '', ''])
    ax[i].tick_params(size=tick_size, width=tick_width)
    ax[i].xaxis.set_ticks([0, 0.5, 1.0])
    # ax[i].xaxis.set_ticklabels(['', '', ''])
    ax[i].grid(True, alpha=0.1)

# Add sample size annotation
# fig.text(0.5, 0.02, f'Oskar-GFP Transgenic (WT Oskar-HA-bcd3UTR) | N={n_samples} images', 
#          ha='center', fontsize=font_size+1)

plt.tight_layout()
# Save output
plt.savefig('C:\\Users\\anr612\\Downloads\\LOTUS-MS\\GFPOskar\\linescans_'+date+"_"+cell+"_"+tissue_stain+"_oskagfpwtoskha-Zscore.svg", format='svg', dpi=300)
plt.show()
