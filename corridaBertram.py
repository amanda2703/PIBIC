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
# Teste
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
Jant = np.empty_like(t)
JF1F0 = np.empty_like(t)
Jpdh = np.empty_like(t)
Jo = np.empty_like(t)
Juni = np.empty_like(t)
JNaCa = np.empty_like(t)
JHres = np.empty_like(t)
JHatp = np.empty_like(t)
JHleak = np.empty_like(t)

ADPm[0] = z0[0]
NADHm[0] = z0[1]
Cam[0] = z0[2]
delta_psi[0] = z0[3]
Jant[0] = find_Jant(ADPm[0],delta_psi[0],parametros)
JF1F0[0] =  find_JF1F0(ADPm[0],delta_psi[0],parametros,u[inputs.get("oligo")])
Jpdh[0] = find_Jpdh(u[inputs.get("fbp")],NADHm[0],Cam[0],parametros)
Jo[0] = find_JO(NADHm[0],delta_psi[0],parametros)
Juni[0] = find_Juni(0.1,delta_psi[0],parametros) 
JNaCa[0] = find_JNaCa(Cam[0],0.1,delta_psi[0],parametros)
JHres[0] = find_JHres(NADHm[0],delta_psi[0],parametros)
JHatp[0] = find_JHatp(ADPm[0],delta_psi[0],parametros,u[inputs.get("oligo")])
JHleak[0] = find_JHleak(delta_psi[0],parametros,u[inputs.get("fccp")])

# Solução do sistema de EDO's 
for i in range(1,N):
	tspan = [t[i-1],t[i]]
	z = odeint(modelo,z0,tspan,args= (parametros,u,))
	ADPm[i] = z[1][0]
	NADHm[i] = z[1][1]
	Cam[i] = z[1][2]
	delta_psi[i] = z[1][3]
	Jant[i] = find_Jant(ADPm[i],delta_psi[i],parametros)
	JF1F0[i] =  find_JF1F0(ADPm[i],delta_psi[i],parametros,u[inputs.get("oligo")])
	Jpdh[i] = find_Jpdh(u[inputs.get("fbp")],NADHm[i],Cam[i],parametros)
	Jo[i] = find_JO(NADHm[i],delta_psi[i],parametros)
	Juni[i] = find_Juni(0.1,delta_psi[i],parametros) 
	JNaCa[i] = find_JNaCa(Cam[i],0.1,delta_psi[i],parametros)
	JHres[i] = find_JHres(NADHm[i],delta_psi[i],parametros)
	JHatp[i] = find_JHatp(ADPm[i],delta_psi[i],parametros,u[inputs.get("oligo")])
	JHleak[i] = find_JHleak(delta_psi[i],parametros,u[inputs.get("fccp")])
	z0 = z[1]
	u = tscrypt(t[i], u)

# Gráficos
t_n = t*(1/60000)

plt.figure(1)
plt.plot(t_n,ADPm,'lightcoral',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('ADPm (mM)',fontsize=15)
plt.savefig('ADPm.png', format='png')

plt.figure(2)
plt.plot(t_n,NADHm,'b',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('NADH (mM)',fontsize=15)
plt.savefig('NADHm.png', format='png')

plt.figure(3)
plt.plot(t_n,Cam,'skyblue',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Cam (uM)',fontsize=15)
plt.savefig('Cam.png', format='png')

plt.figure(4)
plt.plot(t_n,delta_psi,'gold',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('dp (mV)',fontsize=15)
plt.savefig('dp.png', format='png')

plt.figure(5)
plt.plot(t_n,Jant,'teal',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jant (uM/ms)',fontsize=15)
plt.savefig('Jant.png', format='png')

plt.figure(6)
plt.plot(t_n,JF1F0,'darkmagenta',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JF1F0 (uM/ms)',fontsize=15)
plt.savefig('JF1F0.png', format='png')

plt.figure(7)
plt.plot(t_n,Jo,'r',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jo (uM/ms)',fontsize=15)
plt.savefig('Jo.png', format='png')

plt.figure(8)
plt.plot(t_n,Juni,'seagreen',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Juni (uM/ms)',fontsize=15)
plt.savefig('Juni.png', format='png')

plt.figure(9)
plt.plot(t_n,JNaCa,'chocolate',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JNaCa (uM/ms)',fontsize=15)
plt.savefig('JNaCa.png', format='png')

plt.figure(10)
plt.plot(t_n,JHres,'palevioletred',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JHres (uM/ms)',fontsize=15)
plt.savefig('JHres.png', format='png')

plt.figure(11)
plt.plot(t_n,JHatp,'limegreen',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JHatp (uM/ms)',fontsize=15)
plt.savefig('JHatp.png', format='png')

plt.figure(12)
plt.plot(t_n,JHleak,'steelblue',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JHleak (uM/ms)',fontsize=15)
plt.savefig('JHleak.png', format='png')
