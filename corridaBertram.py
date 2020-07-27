# Simulação de oximetria baseada no trabalho de Bertram, Pedersenb, Lucianic, Sherman
# A simplified model for mitochondrial ATP production (DOI: 10.1016/j.jtbi.2006.07.019)
# Código referente ao projeto de iniciação científica dos estudantes: Amanda dos Santos Pereira, Elysa Beatriz de Oliveira Damas e Iago Cossentino de Andrade
# Orientador: Jair Trapé Goulart 
# Instituição: Universidade de Brasília (UnB)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# Parâmetros fixos
parametros = {'p1':400,'p2':1,'p3':0.01,'p4':0.6,'p5':0.1,'p6':177,'p7':5,'p8':7}
aux1 = {'p9':0.1,'p10':177,'p11':5,'p12':120,'p13':10,'p14':190,'p15':8.5,'p16':35}
aux2 = {'p19':0.35,'p20':2,'p21':0.01,'p22':1.1,'p23':0.001,'p24':0.016}
aux3 = {'Amtot':15,'kGPDH':0.0005,'NADmtot':10,'gamma':0.001,'fm':0.01,'Cmito':1.8,'gH':0.002,'deltapH':-0.6}
parametros.update(aux1) 
parametros.update(aux2)
parametros.update(aux3)

# Fluxos
def find_Jant(ADPm,delta_psi,parametros):
    FRT = (96480)/(310.16*8315)
    Amtot = parametros['Amtot']
    p19 = parametros['p19']                                                     
    p20 = parametros['p20']
    ATPm = Amtot-ADPm
    RATm = ATPm/ADPm
    Jant = p19*(RATm/(RATm+p20))*(np.exp(0.5*FRT*delta_psi))
    return Jant
 
def find_JF1F0(ADPm,delta_psi,parametros,oligo):
    Amtot = parametros['Amtot']
    p13 = parametros['p13']
    p14 = parametros['p14']
    p15 = parametros['p15']
    p16 = parametros['p16']
    ATPm = Amtot-ADPm
    JF1F0 = oligo*(p13/(p13+ATPm))*(p16/(1+(np.exp((p14-delta_psi)/p15))))
    return JF1F0

def find_Jpdh(FBP,NADHm,Cam,parametros):
    kGPDH = parametros['kGPDH']
    NADmtot = parametros['NADmtot']
    p1 = parametros['p1']
    p2 = parametros['p2']
    p3 = parametros['p3']
    NADm = NADmtot-NADHm
    Jgpdh = kGPDH*np.sqrt(FBP)
    Jpdh = (p1/(p2+(NADHm/NADm)))*(Cam/(p3+Cam))*Jgpdh
    return Jpdh

def find_JO(NADHm,delta_psi,parametros):
    p4 = parametros['p4']
    p5 = parametros['p5']
    p6 = parametros['p6']
    p7 = parametros['p7']
    JO = ((p4*NADHm)/(p5+NADHm))*(1/(1+(np.exp((delta_psi-p6)/p7))))
    return JO

def find_Juni(Cac,delta_psi,parametros):
    p21 = parametros['p21']
    p22 = parametros['p22']
    Juni = ((p21*delta_psi)-p22)*Cac*Cac
    return Juni
 
def find_JNaCa(Cam,Cac,delta_psi,parametros):
    p23 = parametros['p23']
    p24 = parametros['p24']
    JNaCa = p23*(Cam/Cac)*(np.exp(p24*delta_psi))
    return JNaCa
    
def find_Jmito(Cam,Cac,delta_psi,parametros):
    Jmito = find_JNaCa(Cam,Cac,delta_psi,parametros) - find_Juni(Cac,delta_psi,parametros)
    return Jmito
    
def find_JHres(NADHm,delta_psi,parametros):
    p8 = parametros['p8']
    p9 = parametros['p9']
    p10 = parametros['p10']
    p11 = parametros['p11']
    JHres = ((p8*NADHm)/(p9+NADHm))*(1/(1+(np.exp((delta_psi-p10)/p11))))
    return JHres
    
def find_JHatp(ADPm,delta_psi,parametros,oligo):
    Amtot = parametros['Amtot']
    p12 = parametros['p12']
    p13 = parametros['p13']
    p14 = parametros['p14']
    p15 = parametros['p15']
    ATPm = Amtot-ADPm
    JHatp = oligo*(p13/(p13+ATPm))*(p12/(1+(np.exp((p14-delta_psi)/p15))))
    return JHatp
    
def find_JHleak(delta_psi,parametros,fccp):
    gH = parametros['gH']
    deltapH = parametros['deltapH']
    FRT = (96480)/(310.16*8315)
    JHleak = fccp*gH*(delta_psi+(deltapH/FRT))
    return JHleak
    
# Modelo 
def modelo(ci,t,parametros,u):
    gamma = parametros['gamma']
    fm = parametros['fm']
    Cmito = parametros['Cmito']
    Cac = 0.1
    ADPm = ci[0]
    NADHm = ci[1]
    Cam = ci[2]
    delta_psi = ci[3] 
    adpm = gamma*((find_Jant(ADPm,delta_psi,parametros))-(find_JF1F0(ADPm,delta_psi,parametros,u[inputs.get("oligo")])))
    nadhm = gamma*((find_Jpdh(u[inputs.get("fbp")],NADHm,Cam,parametros))-(find_JO(NADHm,delta_psi,parametros)))
    cam = (-1)*fm*(find_Jmito(Cam,Cac,delta_psi,parametros))
    deltapsi = ((find_JHres(NADHm,delta_psi,parametros))-
    ((find_JHatp(ADPm,delta_psi,parametros,u[inputs.get("oligo")]))+(find_Jant(ADPm,delta_psi,parametros))+
    (find_JHleak(delta_psi,parametros,u[inputs.get("fccp")]))+(find_JNaCa(Cam,Cac,delta_psi,parametros))+
    (2*(find_Juni(Cac,delta_psi,parametros)))))/Cmito
    dzdt = [adpm,nadhm,cam,deltapsi]
    return dzdt
    
# Entradas 
inputs = {   

    "fbp": 0,
    "oligo": 1,
    "fccp": 2
    
}

# Inicialmente
u = {};
u[inputs.get("fbp")] = 1.0
u[inputs.get("oligo")] = 1.0
u[inputs.get("fccp")] = 1.0

# Função que determina o tempo de aplicação das entradas
def tscrypt(t, u):

    if(t < 120000):
        u[inputs.get("fbp")] = 1.0
        u[inputs.get("oligo")] = 1.0
        u[inputs.get("fccp")] = 1.0
        
    if(t >= 120000 and t < 360000):
        u[inputs.get("fbp")] = 5.0
        u[inputs.get("oligo")] = 1.0
        u[inputs.get("fccp")] = 1.0
        
    if(t >= 360000 and t < 480000):
        u[inputs.get("fbp")] = 5.0
        u[inputs.get("oligo")] = 0.06
        u[inputs.get("fccp")] = 1.0
        
    if(t > 480000):
        u[inputs.get("fbp")] = 5.0
        u[inputs.get("oligo")] = 0.06
        u[inputs.get("fccp")] = 10.0   
   
    return u
    
# Tempo 
um_min = 60000                                      # Milisegundos em um minuto
min = 1500                                          # Quantidade de pontos por minuto de simulação
q_min = 10                                          # Quantidade de minutos da simulação
t_i = 0.0                                           # Tempo inicial
t_f = um_min*q_min                                  # Tempo final
N = min*q_min                                       # Número total de pontos                                       
t = np.linspace(t_i,t_f,N)                          # Vetor tempo

# Condições Iniciais 
ADPm = 0.1                                                               
NADHm = 0.6                           
Cam = 0.1                         
delta_psi = 93                          
z0 = [ADPm,NADHm,Cam,delta_psi]                         

# Armazenamento de soluções
ADPm = np.empty_like(t)
NADHm = np.empty_like(t)
Cam = np.empty_like(t)
delta_psi = np.empty_like(t)
Jo = np.empty_like(t)

ADPm[0] = z0[0]
NADHm[0] = z0[1]
Cam[0] = z0[2]
delta_psi[0] = z0[3]
Jo[0] = find_JO(NADHm[0],delta_psi[0],parametros)

# Solução do sistema de EDO's 
for i in range(1,N):
    tspan = [t[i-1],t[i]]
    z = odeint(modelo,z0,tspan,args= (parametros,u,))
    ADPm[i] = z[1][0]
    NADHm[i] = z[1][1]
    Cam[i] = z[1][2]
    delta_psi[i] = z[1][3]
    Jo[i] = find_JO(NADHm[i],delta_psi[i],parametros)
    z0 = z[1]
    u = tscrypt(t[i], u)

t_n = t*(1/60000)

plt.plot(t_n,Jo,'r',linewidth=3.0)
plt.axis([0.1,10.1,0.0,0.42])
plt.xlabel('tempo (min)',fontsize=18)
plt.ylabel('Jo (uM/ms)',fontsize=18)
plt.tick_params(labelsize=12)
plt.show()

plt.plot(t_n,delta_psi,'b',linewidth=3.0)
plt.axis([0.1,10.1,140,190])
plt.xlabel('tempo (min)',fontsize=18)
plt.ylabel('delta_psi (mV)',fontsize=18)
plt.tick_params(labelsize=12)
plt.show()