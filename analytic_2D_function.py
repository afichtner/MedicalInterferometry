import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy import interpolate
from obspy.signal.filter import bandpass

def corr(a,b,maxlag):
	"""
	Correlation with limited lag time.
	a: time series
	b: time series
	maxlag: maximum lag in number of samples
	"""
	n=len(a)
	cc=np.zeros(maxlag)

	for i in range(maxlag): cc[i]=np.sum(a[:n-i]*b[i:])

	return cc



def tt_error(nsrc=300, N=100, L=50.0, freqmin=75.0e3, freqmax=400.0e3):
	"""
	nsrc: number of sources
	N: number of realisations
	L: length of time series in number of transit times
	freqmin: minimum frequency [Hz]
	freqmax: maximum frequency [Hz]
	"""

	# Setup. =============================================

	# Radius of device [m].
	R=0.075
	# Reference velocity for synthetics [m/s].
	c_syn=1500.0
	# Velocity for observations [m/s].
	c_obs=1550.0

	# Random seed for reproducibility.
	#np.random.seed(0)

	# Length of one time series [s].
	T=2.0*L*R/c_syn
	# Number of samples in the time series.
	n=int(10.0*T*freqmax)

	print('transit time: %g ms' % (2000.0*R/c_syn))
	print('total length of time series: %g s' % (N*T))

	# Maximum lag time.
	max_lag_time=2.5*R/c_syn
	df=np.float(n)/T
	print('sampling rate = %g Hz' % df)

	# Source positions.
	phi_src=np.arange(0.0,2.0*np.pi,2.0*np.pi/np.float(nsrc))
	x_src=1.1*R*np.cos(phi_src)
	y_src=1.1*R*np.sin(phi_src)

	# Receiver positions.
	phi_rec=np.arange(0.0,2.0*np.pi,np.pi)
	x_rec=R*np.cos(phi_rec)
	y_rec=R*np.sin(phi_rec)

	# Time increment.
	dt=T/np.float(n)
	# Maximum lag index.
	maxlag=int(max_lag_time/dt)

	# Number of sources.
	nsrc=len(phi_src)
	# Number of receivers.
	nrec=len(phi_rec)

	# Correlations. ======================================

	# Linear correlation stack.
	cc_syn=np.zeros((nrec,maxlag))
	cc_obs=np.zeros((nrec,maxlag))
	# Phase-weights for phase-weighted stack.
	pw_syn=np.zeros((nrec,maxlag),dtype='complex128')
	pw_obs=np.zeros((nrec,maxlag),dtype='complex128')

	# Loop over realisation.
	for i in range(N):
	    
	    u_syn=np.zeros((nrec,n))
	    u_obs=np.zeros((nrec,n))
	    
	    # Make random wavefield.

	    # Loop over sources.
	    for isrc in range(nsrc):

	        # Time series for one source.
	        srcwvlt=0.5-np.random.rand(n)
	        
	        # Loop over receivers.
	        for irec in range(nrec):

	            # Source-receiver distance.
	            r=np.sqrt((x_rec[irec]-x_src[isrc])**2+(y_rec[irec]-y_src[isrc])**2)
	            # Time shift.
	            shift_t_syn=r/c_syn
	            shift_t_obs=r/c_obs
	            # Index shift.
	            shift_n_syn=int(np.round(shift_t_syn/dt))
	            shift_n_obs=int(np.round(shift_t_obs/dt))
	            # Assign shifted time series to receiver.
	            u_syn[irec,shift_n_syn:]+=srcwvlt[0:n-shift_n_syn]/np.sqrt(r)
	            u_obs[irec,shift_n_obs:]+=srcwvlt[0:n-shift_n_obs]/np.sqrt(r)
	            
	    # Compute and stack correlations.
	    for irec in range(1,nrec):
	        # Correlation.
	        cc_i_syn=corr(u_syn[0,:],u_syn[irec,:],maxlag)
	        cc_i_obs=corr(u_obs[0,:],u_obs[irec,:],maxlag)
	        # Bandpass.
	        cc_i_syn=bandpass(cc_i_syn,freqmin=freqmin,freqmax=freqmax,df=df,corners=4,zerophase=True)
	        cc_i_obs=bandpass(cc_i_obs,freqmin=freqmin,freqmax=freqmax,df=df,corners=4,zerophase=True)
	        # Linear stack.
	        cc_syn[irec,:]+=cc_i_syn
	        cc_obs[irec,:]+=cc_i_obs
	        # Phase weights.
	        h_syn=ss.hilbert(cc_i_syn)
	        h_obs=ss.hilbert(cc_i_obs)
	        pw_syn[irec,:]+=h_syn/np.abs(h_syn)
	        pw_obs[irec,:]+=h_obs/np.abs(h_obs)
	        
	# Normalise.
	cc_syn=cc_syn/np.max(cc_syn)
	cc_obs=cc_obs/np.max(cc_obs)
	pw_syn=np.abs(pw_syn/np.max(pw_syn))
	pw_obs=np.abs(pw_obs/np.max(pw_obs))

	# Compute time shifts and return. ===================

	cc_syn_pw=(pw_syn[irec,:]**2.0)*cc_syn[1,:]
	cc_obs_pw=(pw_obs[irec,:]**2.0)*cc_obs[1,:]

	irec=1

	# Predicted traveltimes between receiver 0 and receiver irec.
	tt_syn=1000*np.sqrt((x_rec[0]-x_rec[irec])**2+(y_rec[0]-y_rec[irec])**2)/c_syn
	tt_obs=1000*np.sqrt((x_rec[0]-x_rec[irec])**2+(y_rec[0]-y_rec[irec])**2)/c_obs

	# Analytical traveltime difference. =====================================

	dtt_analytic=tt_syn-tt_obs

	# Traveltime difference by waveform correlation. ========================

	# Interpolate to increase time resolution.
	t=1000.0*np.linspace(0.0,np.float(maxlag)*dt,maxlag)
	f_syn=interpolate.interp1d(t,cc_syn_pw,kind='cubic')
	f_obs=interpolate.interp1d(t,cc_obs_pw,kind='cubic')

	t_interp=1000.0*np.linspace(0.0,np.float(maxlag)*dt,50*maxlag)
	cc_syn_interp=f_syn(t_interp)
	cc_obs_interp=f_obs(t_interp)

	# Define measurement window.
	T_win=0.5/freqmin
	t_min=tt_syn-2000.0*T_win/0.2
	t_max=tt_syn+2000.0*T_win/0.2
	win=np.array([t_interp>t_min]).astype(float)*np.array([t_interp<t_max]).astype(float)
	win=win[0]

	n_shift=int(2.0*np.abs((tt_syn-tt_obs)/(t_interp[1]-t_interp[0])))
	cc=corr(win*cc_obs_interp,win*cc_syn_interp,n_shift)
	tt=np.linspace(0.0,n_shift*(t_interp[1]-t_interp[0]),n_shift)

	dtt_numeric=np.float(tt[np.max(cc)==cc])

	# Return
	ddtt=dtt_analytic-dtt_numeric
	ddtt_rel=100.0*ddtt/tt_syn
	dv=-c_syn*ddtt/tt_syn

	return ddtt, ddtt_rel, dv
	

