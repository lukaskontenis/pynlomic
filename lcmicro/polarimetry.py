
"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This module contains polarimetry routines

Some ideas are taken from my collection of Matlab scripts developed while in
the Barzda lab at the University of Toronto in 2011-2017.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
from lklib.util import isstring, find_closest
from numpy import zeros, sin, cos, arcsin, arccos, pi, matrix, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def RotateMuellerMatrix( M, theta ):
    """
    Rotate an element with Mueller matrix 'M' by angle 'theta'.
    """
    
    Mr = zeros( [ 4, 4 ] );
    
    Mr[ 0, 0 ] = 1;
    
    Mr[ 1, 1 ] = cos( 2*theta );
    Mr[ 1, 2 ] = -sin( 2*theta );
    Mr[ 2, 1 ] = sin( 2*theta );
    Mr[ 2, 2 ] = cos( 2*theta );
    
    Mr[ 3, 3 ] = 1;
    
    return Mr * M * Mr.transpose();

def GetMuellerMatrix( element, theta, *args ):
    """
    Get the Muller matrix of 'element' rotated at angle theta. q,r are
    diatenuation coefficients, only required for 'PolD' element.
    """

    M = zeros( [ 4, 4 ] );
    
    if( element == "HWP" ):
        M[0,0] = 1;
        M[1,1] = 1;
        M[2,2] = -1;
        M[3,3] = -1;
        
    elif( element == "QWP" ):
        M[0,0] = 1
        M[1,1] = 1
        M[2,3] = 1
        M[3,2] = -1
        
    elif( element == "RTD" or element == "Retarder"):
        d = args[0]
        c = cos(d);
        s = sin(d);
        
        M[0,0] = 1
        M[1,1] = 1
        M[2,2] = c
        M[2,3] = s
        M[3,2] = -s
        M[3,3] = c
        
    elif( element == "POL" or element == "Polarizer" ):
        
        M[0,0] = 1
        M[0,1] = 1
        M[1,0] = 1
        M[1,1] = 1
        M = M*0.5;
        
    elif( element == "NOP" or element == "Empty" ):
        
        M[0,0] = 1
        M[1,1] = 1
        M[2,2] = 1
        M[3,3] = 1
        
    else:
        print( "Element ''%s'' not defined", element )
        
    M = matrix(M)

    M = RotateMuellerMatrix( M, theta );
    
    return M


def GetStokesVector( state ):
    """
    Get the Stokes vector of 'state'.
    """
    
    S = zeros( [4,1] );
        
    if( not isstring( state )):
        # Linear polarizations. Gamma happens to be equal to LP orientation 
        gamma = state / 180 * pi;
        omega = 0;
    else:
        if( state == "HLP" ):
            gamma = 0;
            omega = 0;
            
        elif( state == "VLP" ):
            gamma = pi/2;
            omega = 0;
            
        elif( state == "RCP" ):
            gamma = 0;
            omega = pi/4;
            
        elif( state == "LCP" ):
            gamma = 0;
            omega = -pi/4;
                
        else:
            print( 'State ''%s'' not defined' %state );
    
    S[0] = 1
    S[1] = cos( 2*gamma ) * cos( 2*omega )
    S[2] = sin( 2*gamma ) * cos( 2*omega )
    S[3] = sin( 2*omega )
    S = matrix(S)
    
    return S


def TestPolarizerTransmission():
    Sin = GetStokesVector( "HLP" )
    
    th_arr = np.arange( 0, pi, pi/100 )
    I = np.zeros_like( th_arr )

    for ( ind, th ) in enumerate( th_arr ):
        M_pol = GetMuellerMatrix( "POL", th )
        Sout = M_pol * Sin
        I[ ind ] = Sout[0]
        
    plt.plot( th_arr/pi*180, I )
    
def BerekGetTiltByGearAngle( a ):
    return 0.000073 * a**3 -0.001763 * a**2 +0.068106 * a -0.028374

def GetMgF2_no( L ):
    
    L2 = L**2
    
    return sqrt( 1 + 0.48755108*L2/(L2 - 0.04338408**2) + 0.39875031*L2/(L2-0.09461442**2) + 2.3120353*L2/(L2-23.793604**2 ))

def GetMgF2_ne( L ):
    
    L2 = L**2
    
    return sqrt( 1 + 0.41344023*L2/(L2-0.03684262**2) + 0.50497499*L2/(L2-0.09076162**2) + 2.4904862*L2/(L2-23.771995**2 ))


def BerekGetRetardationByTilt( th_I, lam = 1.03 ):
    
    # Crystal thickness
    L_0 = 2000
    
    # Wavelength
    #lam = 1.03
    
    # MgF2 refractive indices
    ne = GetMgF2_ne( 1.3850 )
    no = GetMgF2_no( 1.3734 )
    
    # Refractive index of air
    n1 = 1.00027408
    
    # Average refractive index of crystal. Only retardation due to phase delay
    # in the crystal is considered, refraction angle birefringence is ignored.
    n2 = (ne+no)/2
    
    # Incident angle
    #th_I = pi/2 - a
    
    # Refracted angle in crystal
    # IGNORING: refraction birefringence, n2 varies with angle of incidence
    sin_th_R = n1/n2 * sin( th_I )
    cos_th_R = np.sqrt( 1 - sin_th_R**2 )
    
    # Internal first refractive index in the crystal - no for uniaxial
    nI1 = no
    
    # Internal second refractive index in the crystal - varies based on the
    # angle of propagation in the crystal. nI2 = no when th_R = 0 and nI1 = ne
    # when th_R = pi/2
    nI2 = no*ne / np.sqrt( no**2 * sin_th_R**2 + ne**2 * cos_th_R**2 )
    
    # Propagation path length in the tilted crystal
    L = L_0 / cos_th_R
    
    # Phase difference in rad
    return ( nI2 - nI1 ) * 2*pi/lam * L

def PlotBerekTransmissionPanels_all():
    plt.clf()
    pol_th_arr = np.arange( 0, 121, 30 )/180*pi
    #pol_th_arr = np.arange( 0, 5, 1 )/180*pi
    #pol_th_arr = np.arange( 0, , 45 )/180*pi
    lam_arr = [ 0.45, 0.515, 0.76, 1.03 ]
    
    for ( ind_lam, lam ) in enumerate( lam_arr ):
        for ( ind_pol_th, pol_th ) in enumerate( pol_th_arr ):
            ax = plt.subplot( 4, 5, ind_lam*5 + ind_pol_th + 1 )
            
            if( ind_lam == len( lam_arr )-1 ):
                bottom_row = True
            else:
                bottom_row = False
                
            if( ind_lam == 0 ):
                top_row = True
            else:
                top_row = False
                
            if( ind_pol_th == 0 ):
                left_col = True
            else:
                left_col = False
            
            [ I, X, Y ] = GetBerekTransmissionMap( lam = lam, pol_th = pol_th )
            PlotBerekTransmissionMapPanel( ax, I = I, X = X, Y = Y, lam = lam, pol_th = pol_th,
                                       bottom_row = bottom_row, top_row = top_row,
                                       left_col = left_col )
            
            
def PlotBerekTransmissionPanels_diff( Meas = None, Meas_lam = None, Meas_pol_th = None ):
    plt.clf()
    pol_th_arr = np.arange( 0, 121, 30 )/180*pi
    #pol_th_arr = np.arange( 0, 5, 1 )/180*pi
    #pol_th_arr = np.arange( 0, , 45 )/180*pi
    lam_arr = [ 0.45, 0.515, 0.76, 1.03 ]
    
    for ( ind_lam, lam ) in enumerate( lam_arr ):
        for ( ind_pol_th, pol_th ) in enumerate( pol_th_arr ):
            ax = plt.subplot( 4, 5, ind_lam*5 + ind_pol_th + 1 )
            
            if( ind_lam == len( lam_arr )-1 ):
                bottom_row = True
            else:
                bottom_row = False
                
            if( ind_lam == 0 ):
                top_row = True
            else:
                top_row = False
                
            if( ind_pol_th == 0 ):
                left_col = True
            else:
                left_col = False
            
            [ I, X, Y ] = GetBerekTransmissionMap( lam = lam, pol_th = pol_th )
            D = Meas[:,:,Meas_lam.index(lam), Meas_pol_th.index(pol_th) ]
            PlotBerekTransmissionMapPanel( ax, I = D, X = X, Y = Y, lam = lam, pol_th = pol_th,
                                       bottom_row = bottom_row, top_row = top_row,
                                       left_col = left_col )
            
 
               
def PlotBerekTransmissionMapPanel( ax, I = None, X = None, Y = None, lam = 0, pol_th = 0, bottom_row = False, top_row = False, left_col = False ):
    PlotBerekTransmissionMap( I = I, X = X, Y = Y )
    plt.title( '%d nm, LP %d deg' %( lam*1000, pol_th/pi*180 ))
    
    ax = plt.gca()
    
    if( bottom_row == False ):
        ax.set_xlabel('')
        ax.set_xticklabels([])
        
    if( top_row == False or left_col == False ):
        ax.set_ylabel('')
   

def PlotBerekTransmissionMap( I, X, Y, grid_step = 90 ):
    
    X_deg = X/pi*180
    Y_deg = Y/pi*180
    
    plt.imshow( I, origin='lower',
               extent=[ X_deg.min(), X_deg.max(), Y_deg.min(), Y_deg.max() ],
               clim=[0,1] )
    
    ax = plt.gca()
    plt.grid('on')
    #ax.set_aspect(360/(y_max-y_min))
    
    plt.xticks( np.arange( X_deg.min(), X_deg.max(), grid_step) )
    
    plt.xlabel( 'Rotation, deg' )
    
    plt.ylabel( 'Tilt gear rotation, deg' )
    plt.xticks( np.arange( Y_deg.min(), Y_deg.max(), grid_step) )
        
    #    ax.twinx()
    #    rtd_ax_from = round( rtd_arr[0]/2/pi )
    #    rtd_ax_to = round( rtd_arr[-1]/2/pi )
    #    rtd_ax_step = np.sign(rtd_ax_to - rtd_ax_from)*0.5
    #    
    #    plt.ylim( [ rtd_arr[0]/2/pi, rtd_arr[-1] /2/pi ] )
    #    plt.yticks( np.arange( rtd_ax_from, rtd_ax_to+rtd_ax_step, rtd_ax_step ))           
                    
        
    
def GetBerekTransmissionMap( lam = 1.03, pol_th = 0, Npts = 35 + 1 ):
    #Npts = 35*2 + 1
    
    Sin = GetStokesVector( "VLP" )
    M_qwp = GetMuellerMatrix( "QWP", 10/180*pi )
    
    th_arr = np.arange( 0, 351/180*pi, 350/180*pi/(Npts-1) )

    # Waveplate tilting gear rotation in radians
    y_min = 0
    y_max = 350/180*pi
    tgr_arr = np.arange( 0, 351/180*pi, 350/180*pi/(Npts-1) )
    
    tlt_arr = BerekGetTiltByGearAngle( tgr_arr )
    rtd_arr = BerekGetRetardationByTilt( tlt_arr, lam )
        
    I = np.zeros( [ len(tgr_arr), len(th_arr) ] )
    
    M_pol = GetMuellerMatrix( "POL", pol_th )

    for ( ind_rtd, rtd ) in enumerate( rtd_arr ):
        for ( ind_th, th ) in enumerate( th_arr ):
            M_brk = GetMuellerMatrix( "RTD", th, rtd )
            
            Sout = M_pol * M_brk * M_qwp * Sin
            
            I[ ind_rtd, ind_th ] = Sout[0]
            
    return ( I, th_arr, tgr_arr )
#    
#    plt.imshow( I.transpose(), origin='lower', extent=[0,360,y_min,y_max/pi*180], clim=[0,1] )
#    #plt.imshow( I.transpose(), origin='lower', extent=[0,360,y_min,y_max] )
#    
#    ax = plt.gca()
#    plt.grid('on')
#    #ax.set_aspect(360/(y_max-y_min))
#    
#    plt.xticks( np.arange(0,361,grid_step) )
#    
#    plt.xlabel( 'Rotation, deg' )
#    
#    plt.ylabel( 'Tilt gear rotation, deg' )
#    plt.yticks( np.arange(0,361,grid_step) )
        
#    ax.twinx()
#    rtd_ax_from = round( rtd_arr[0]/2/pi )
#    rtd_ax_to = round( rtd_arr[-1]/2/pi )
#    rtd_ax_step = np.sign(rtd_ax_to - rtd_ax_from)*0.5
#    
#    plt.ylim( [ rtd_arr[0]/2/pi, rtd_arr[-1] /2/pi ] )
#    plt.yticks( np.arange( rtd_ax_from, rtd_ax_to+rtd_ax_step, rtd_ax_step ))
 
       
def PlotBerekTiltGearCalib():
    gear_a = np.arange( 0, 2*pi, 2*pi/200 )
    
    tilt_a = BerekGetTiltByGearAngle( gear_a )
    
    plt.plot( gear_a, tilt_a )
    
    plt.xlabel( "Gear angle, rad" )
    plt.ylabel( "Tilt angle, rad" )
    plt.grid('on')
    
def PlotBerekTiltRetardanceCalib():
    tgr_arr = np.arange( 0, 2*pi, 2*pi/200 )
    
    tlt_arr = BerekGetTiltByGearAngle( tgr_arr )
    
    rtd_arr = BerekGetRetardationByTilt( tlt_arr )
    
    plt.plot( tgr_arr, rtd_arr )
    
    plt.xlabel( "Gear angle, rad" )
    plt.ylabel( "Retardabce, rad" )
    plt.grid('on')

    
def CompareBerekCalib( Meas, Meas_lam, Meas_pol_th ):
    PlotBerekTransmissionMapPanel
    
    num_sim_pts = 35*2 + 1
    
    pol_th_arr = [ 0, 30, 60, 90, 120 ]
    lam = 1.03
    
    X = np.arange(0,351,10)
    Y = np.arange(0,351,10)
    
    Xs = np.arange( 0, 351/180*pi, 350/180*pi/(num_sim_pts-1) )
    Ys = np.arange( 0, 351/180*pi, 350/180*pi/(num_sim_pts-1) )
    
    ind_row = 0
    for ( ind_pol_th, pol_th ) in enumerate( pol_th_arr ):
        ax = plt.subplot( 4, len( pol_th_arr ), ind_row*len( pol_th_arr ) + ind_pol_th + 1 )
        
        PlotBerekTransmissionMapPanel(
                ax, 
                lam = lam,
                pol_th = pol_th,
                I = Meas[:,:, Meas_lam.index(lam), Meas_pol_th.index( pol_th )],
                X = X,
                Y = Y )
        
    Sim = np.ndarray( [num_sim_pts,num_sim_pts,len(pol_th_arr)] )
        
    ind_row = 1
    for ( ind_pol_th, pol_th ) in enumerate( pol_th_arr ):
        ax = plt.subplot( 4, len( pol_th_arr ), ind_row*len( pol_th_arr ) + ind_pol_th + 1 )
        
        I = GetBerekTransmissionMap( lam = lam, pol_th = pol_th )
        Sim[:,:,ind_pol_th ] = I
        
        PlotBerekTransmissionMapPanel(
                ax, 
                lam = lam,
                pol_th = pol_th,
                I = I,
                X = X,
                Y = Y )
        
    ind_row = 2
    tilt = 150
    for ( ind_pol_th, pol_th ) in enumerate( pol_th_arr ):
        ax = plt.subplot( 4, len( pol_th_arr ), ind_row*len( pol_th_arr ) + ind_pol_th + 1 )
        
        sim_ind = find_closest( Xs, tilt/180*pi )
        meas_ind = find_closest( X, tilt )
        
        plt.plot( Xs/pi*180, Sim[sim_ind,:,ind_pol_th ],
                 X, Meas[meas_ind,:, Meas_lam.index(lam), Meas_pol_th.index( pol_th )] )
        