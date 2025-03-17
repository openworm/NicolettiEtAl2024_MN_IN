# "Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families"
# M. Nicoletti et al. PloS ONE, 19(3): e0298105.
# https://doi.org/10.1371/journal.pone.0298105

def AIY_simulation_iclamp(gAIY_scaled,s1,s2,ns, delay = 1000, duration = 5000, simdur = 7000, transient = 4900, V_init = -45):
    
 
    from neuron import h,gui
    import numpy
    import math
    from operator import add

    surf=65.89e-8 # surface in cm^2 form neuromorpho AIYL
    vol=7.42e-12 # total volume
    L=math.sqrt(surf/math.pi)
    rsoma=L*1e4
    cm_uFcm2=gAIY_scaled[8]
 
    
    
    soma=h.Section(name="soma")
    soma.L=rsoma
    soma.diam=rsoma
    soma.cm=cm_uFcm2
    soma.Ra=100
    h.psection(sec=soma)
    
    soma.insert('egl19')
    soma.insert('slo1egl19')
    
    soma.insert('nca')
    soma.insert('leak')
    soma.insert('slo1iso')
    soma.insert('kqt1')
    soma.insert('shl1')

    
    
    
    for seg in soma:
        
        seg.leak.gbar =  gAIY_scaled[0]
        seg.slo1iso.gbar =  gAIY_scaled[1]
        seg.kqt1.gbar=gAIY_scaled[2]                
        seg.egl19.gbar=gAIY_scaled[3]
        seg.slo1egl19.gbar = gAIY_scaled[4]        
        seg.nca.gbar =  gAIY_scaled[5]
        seg.shl1.gbar =  gAIY_scaled[6]        
        seg.leak.e=gAIY_scaled[7]
        
    
        seg.eca=60
        seg.ek=-80

    
    stim=h.IClamp(soma(0.5))
    dir(stim)
    
    stim.delay=delay
    stim.amp=10
    stim.dur=duration
    
    v_vec = h.Vector()   
    cai_vec = h.Vector()
    t_vec = h.Vector()        # Time stamp vector
    v_vec.record(soma(0.5)._ref_v)
    cai_vec.record(soma(0.5)._ref_cai)
    t_vec.record(h._ref_t)

    simdur =simdur

    ref_v=[]
    ref_cai = []
    ref_t=[]

    print("All parameters used in current clamp:")
    h.psection(sec=soma)
    h("eca")
    
    for i in numpy.linspace(start=s1, stop=s2, num=ns):
        
         stim.amp=i
         h.tstop=simdur
         h.dt=0.4
         h.finitialize(-60)
         h.run()
            
         ref_t_vec=numpy.zeros_like(t_vec)
         t_vec.to_python(ref_t_vec)
         ref_t.append(ref_t_vec)
            
        
         ref_v_vec=numpy.zeros_like(v_vec)
         v_vec.to_python(ref_v_vec)
         ref_v.append(ref_v_vec)

         ref_cai_vec=numpy.zeros_like(cai_vec)
         cai_vec.to_python(ref_cai_vec) 
         ref_cai.append(ref_cai_vec)
            
            
    v=[]
    v=numpy.array(list(ref_v))
    ca1=numpy.array(list(ref_cai))
    time1=numpy.array(ref_t)
    
   
    
    # ## SS VOLTAGE-CURRENT RELATION
    # ind=numpy.where(numpy.logical_and(time1[0]>=transient, time1[0]<=transient))
    # ind_max=numpy.amax(ind)
    # ind_min=numpy.amin(ind)
    # vi=numpy.mean(v[:,ind_min:ind_max],axis=1)
	
	#  # PEAK VOLTAGE-CURRENT RELATION
    # ind2=numpy.where(numpy.logical_and(time1[0]>=1000, time1[0]<=transient))
    # ind2_max=numpy.amax(ind2)
    # ind2_min=numpy.amin(ind2)
    # vi_peak=numpy.amax(v[:,ind2_min:ind2_max])
    # vi_peak=[]

    

    
    # for j in range(ns):
    #     if j<=2:
    #         peak=numpy.amin(v[j,ind2_min:ind2_max])
    #     else:
    #         peak=numpy.amax(v[j,ind2_min:ind2_max])
    #     vi_peak.append(peak)

    return v, time1, ca1 
    
    
    
        


