# Simulação de oximetria baseada nos trabalhos de Gerhard Magnus e Joel Keizer:
# 1) Minimal model of b-cell mitochondrial Ca2+ handling (DOI: 10.1152/ajpcell.1997.273.2.C717)
# 2) Model of b-cell mitochondrial calcium handling and electrical activity. I. Cytoplasmic variables (DOI: 10.1152/ajpcell.1998.274.4.C1158)
# Código referente ao projeto de iniciação científica dos estudantes: Amanda dos Santos Pereira, Elysa Beatriz de Oliveira Damas e Iago Cossentino de Andrade
# Orientador: Jair Trapé Goulart 
# Instituição: Universidade de Brasília (UnB)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parâmetros tabelados --------------------------------------------------------------------------------------------------------------------------------------------------
outros = {'F':96485,'R':8314,'NADtot':8,'Atot':12,'Atoti':1.5}    
tabela1 = {'deltapH':-0.4,'T':310,'gH':0.2}
tabela2 = {'Kres':1.35e18,'pres':0.4,'r1':2.077e-18,'r2':1.728e-9,'r3':1.059e-26,'ra':6.394e-10,'rb':1.762e-13,'rc1':2.656e-19,'rc2':8.632e-27,'dpB':50,'g':0.85}
tabela3 = {'KF1':1.71e6,'Pim':20,'pF1':0.7,'p1':1.346e-8,'p2':7.739e-7,'p3':6.65e-15,'pa':1.656e-5,'pb':3.373e-7,'pc1':9.651e-14,'pc2':4.845e-19,'dpB':50}
tabela4 = {'JmaxANT':1000,'f':0.5}
tabela5 = {'Kcat':6.0,'Kact':0.38,'Jmaxuni':400,'dpuni':91,'nact':2.8,'Lunimax':50,'KNa':9.4,'KCa':0.003,'Nai':30,'dpNaCa':91,'b':0.5,'JmaxNaCa':5.5,'n':3}
betas = {'betamax':126,'beta1':1.66,'beta2':0.0249,'beta3':4,'beta4':2.83,'beta5':1.3,'beta6':2.66,'beta7':0.16}
table3 = {'u1':15,'u2':1.1,'KCa2':0.05,'Jredbasal':20,'CCa2m':3000,'Cmito':1.45e-3,'khyd':41,'tauhyd':50,'dJhydmax':30.1,'kGlc':8.7,'nhyd':2.7}
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# FLUXOS ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def find_Jo(outros,tabela1,tabela2,NADHm,dp):
    F = outros['F']
    R = outros['R']
    NADtot = outros['NADtot']
    deltapH = tabela1['deltapH']
    T = tabela1['T']
    Kres = tabela2['Kres']
    pres = tabela2['pres']
    r1 = tabela2['r1']
    r2 = tabela2['r2']
    r3 = tabela2['r3']
    ra = tabela2['ra']
    rc1 = tabela2['rc1']
    rc2 = tabela2['rc2']
    dpB = tabela2['dpB']
    g = tabela2['g']
    FRT = F/(R*T)
    NADm = NADtot-NADHm
    Ares = ((R*T)/F)*(np.log((Kres*(np.sqrt(NADHm)))/(np.sqrt(NADm))))
    n = (((ra*(np.power(10,(6*deltapH))))+(rc1*(np.exp(6*FRT*dpB))))*(np.exp(FRT*Ares)))-(ra*(np.exp(g*6*FRT*dp)))+(rc2*(np.exp(FRT*Ares))*(np.exp(g*6*FRT*dp)))
    d = ((1+(r1*(np.exp(FRT*Ares))))*(np.exp(6*FRT*dpB)))+((r2+(r3*(np.exp(FRT*Ares))))*(np.exp(g*6*FRT*dp)))
    Jo = 30*pres*(n/d)
    return Jo

def find_JHres(outros,tabela1,tabela2,NADHm,dp):
    F = outros['F']
    R = outros['R']
    NADtot = outros['NADtot']
    deltapH = tabela1['deltapH']
    T = tabela1['T']
    Kres = tabela2['Kres']
    pres = tabela2['pres']
    r1 = tabela2['r1']
    r2 = tabela2['r2']
    r3 = tabela2['r3']
    ra = tabela2['ra']
    rb = tabela2['rb']
    dpB = tabela2['dpB']
    g = tabela2['g']
    FRT = F/(R*T)
    NADm = NADtot-NADHm
    Ares = ((R*T)/F)*(np.log((Kres*(np.sqrt(NADHm)))/(np.sqrt(NADm))))
    n = (ra*(np.power(10,(6*deltapH)))*(np.exp(FRT*Ares)))-((ra+rb)*(np.exp(g*6*FRT*dp)))
    d = ((1+(r1*(np.exp(FRT*Ares))))*(np.exp(6*FRT*dpB)))+((r2+(r3*(np.exp(FRT*Ares))))*(np.exp(g*6*FRT*dp)))
    JHres = 360*pres*(n/d)
    return JHres

def find_JHleak(outros,tabela1,dp,fccp):
    F = outros['F']
    R = outros['R']
    deltapH = tabela1['deltapH']
    T = tabela1['T']
    gH = tabela1['gH']
    Z = 2.303*((R*T)/F)
    deltap = dp-(Z*deltapH)
    JHleak = gH*deltap*fccp
    return JHleak
    
def find_JpF1(outros,tabela1,tabela3,ADPm,dp,oligo):
    F = outros['F']
    R = outros['R']
    Atot = outros['Atot']
    deltapH = tabela1['deltapH']
    T = tabela1['T']
    KF1 = tabela3['KF1']
    Pim = tabela3['Pim']
    pF1 = tabela3['pF1']
    p1 = tabela3['p1']
    p2 = tabela3['p2']
    p3 = tabela3['p3']
    pa = tabela3['pa']
    pc1 = tabela3['pc1']
    pc2 = tabela3['pc2']
    dpB = tabela3['dpB']
    FRT = F/(R*T)
    ATPm = Atot-ADPm
    ADPmfree = 0.8*ADPm 
    AF1 = ((R*T)/F)*(np.log((KF1*ATPm)/(ADPmfree*Pim)))
    n = (((pa*(np.power(10,(3*deltapH))))+(pc1*(np.exp(3*FRT*dpB))))*(np.exp(FRT*AF1)))-(pa*(np.exp(3*FRT*dp)))+(pc2*(np.exp(FRT*AF1))*(np.exp(3*FRT*dp)))
    d = ((1+(p1*(np.exp(FRT*AF1))))*(np.exp(3*FRT*dpB)))+((p2+(p3*(np.exp(FRT*AF1))))*(np.exp(3*FRT*dp)))
    JpF1 = -60*pF1*(n/d)*oligo
    return JpF1
    
def find_JHF1(outros,tabela1,tabela3,ADPm,dp,oligo):
    F = outros['F']
    R = outros['R']
    Atot = outros['Atot']
    deltapH = tabela1['deltapH']
    T = tabela1['T']
    KF1 = tabela3['KF1']
    Pim = tabela3['Pim']
    pF1 = tabela3['pF1']
    p1 = tabela3['p1']
    p2 = tabela3['p2']
    p3 = tabela3['p3']
    pa = tabela3['pa']
    pb = tabela3['pb']
    dpB = tabela3['dpB']
    FRT = F/(R*T)
    ATPm = Atot-ADPm
    ADPmfree = 0.8*ADPm 
    AF1 = ((R*T)/F)*(np.log((KF1*ATPm)/(ADPmfree*Pim)))
    n = (pa*(np.power(10,(3*deltapH)))*(np.exp(FRT*AF1)))-((pa+pb)*(np.exp(3*FRT*dp)))
    d = ((1+(p1*(np.exp(FRT*AF1))))*(np.exp(3*FRT*dpB)))+((p2+(p3*(np.exp(FRT*AF1))))*(np.exp(3*FRT*dp)))
    JHF1 = -180*pF1*(n/d)*oligo
    return JHF1

def find_Jant(outros,tabela1,tabela4,ADPm,ADPi,dp):
    F = outros['F']
    R = outros['R']
    Atot = outros['Atot']
    Atoti = outros['Atoti']
    T = tabela1['T']
    JmaxANT = tabela4['JmaxANT']
    f = tabela4['f']
    FRT = F/(R*T)
    ATPm = Atot-ADPm
    ATPi = Atoti-ADPi
    ATP4m = 0.05*ATPm
    ATP4i = 0.05*ATPi
    ADP3m = 0.36*ADPm
    ADP3i = 0.135*ADPi
    n = 1-((ATP4i/ADP3i)*(ADP3m/ATP4m)*(np.exp(-1*FRT*dp)))
    d = (1+((ATP4i/ADP3i)*(np.exp(-1*f*FRT*dp))))*(1+(ADP3m/ATP4m))
    Jant = JmaxANT*(n/d)
    return Jant   

def find_Jant2(outros,tabela1,tabela4,ADPm,ADPi,total,dp):							
    F = outros['F']
    R = outros['R']
    Atot = outros['Atot']
    T = tabela1['T']
    JmaxANT = tabela4['JmaxANT']
    f = tabela4['f']
    FRT = F/(R*T)
    ATPm = Atot-ADPm
    ATPi = total-ADPi
    ATP4m = 0.05*ATPm
    ATP4i = 0.05*ATPi
    ADP3m = 0.36*ADPm
    ADP3i = 0.135*ADPi
    n = 1-((ATP4i/ADP3i)*(ADP3m/ATP4m)*(np.exp(-1*FRT*dp)))
    d = (1+((ATP4i/ADP3i)*(np.exp(-1*f*FRT*dp))))*(1+(ADP3m/ATP4m))
    Jant = JmaxANT*(n/d)
    return Jant
    
def find_Juni(outros,tabela1,tabela5,Cai,dp):
    F = outros['F']
    R = outros['R']
    T = tabela1['T']
    Kcat = tabela5['Kcat']
    Kact = tabela5['Kact']
    Jmaxuni = tabela5['Jmaxuni']
    dpuni = tabela5['dpuni']
    nact = tabela5['nact']
    Lunimax = tabela5['Lunimax']
    FRT = F/(R*T)
    t1 = ((Cai/Kcat)*(np.power((1+(Cai/Kcat)),3)))/((np.power((1+(Cai/Kcat)),4))+(Lunimax/(np.power((1+(Cai/Kact)),nact))))
    t2 = (2*FRT*(dp-dpuni))/(1-(np.exp(-2*FRT*(dp-dpuni))))
    Juni = t1*Jmaxuni*t2
    return Juni
    
def find_JNaCa(outros,tabela1,tabela5,Cam,dp):
    F = outros['F']
    R = outros['R']
    T = tabela1['T']
    KNa = tabela5['KNa']
    KCa = tabela5['KCa']
    Nai = tabela5['Nai']
    dpNaCa = tabela5['dpNaCa']
    b = tabela5['b']
    JmaxNaCa = tabela5['JmaxNaCa']
    n = tabela5['n']
    FRT = F/(R*T)
    JNaCa = JmaxNaCa*(np.exp(b*FRT*(dp-dpNaCa)))/((np.power((1+(KNa/Nai)),n))*(1+(KCa/Cam)))
    return JNaCa

def find_dJglytotal(betas,ATPi,Glc):
    betamax = betas['betamax']
    beta1 = betas['beta1']
    beta2 = betas['beta2']
    beta3 = betas['beta3']
    beta4 = betas['beta4']
    beta5 = betas['beta5']
    beta6 = betas['beta6']
    beta7 = betas['beta7']
    dJglytotal = (betamax*(1+(beta1*Glc))*beta2*Glc*ATPi)/(1+(beta3*ATPi)+((1+(beta4*ATPi))*beta5*Glc)+((1+(beta6*ATPi))*beta7*Glc*Glc))
    return dJglytotal

def find_fPDH(outros,table3,NADHm,Cam):
	NADtot = outros['NADtot']
	u1 = table3['u1']
	u2 = table3['u2']
	KCa2 = table3['KCa2']
	NADm = NADtot-NADHm
	fPDH = 1/(1+(u2*((1+(u1*(1/(np.power((1+(Cam/KCa2)),2)))))*((NADHm/NADm)+1))))
	return fPDH

def find_Jred(outros,betas,table3,NADHm,ATPi,Cam,Glc):
    Jredbasal = table3['Jredbasal']
    dJglytotal = find_dJglytotal(betas,ATPi,Glc)
    fPDH = find_fPDH(outros,table3,NADHm,Cam)
    Jred = Jredbasal+(7.36*fPDH*dJglytotal)
    return Jred

def find_JpTCA(outros,betas,table3,NADHm,ATPi,Cam,Glc):
    Jredbasal = table3['Jredbasal']
    dJglytotal = find_dJglytotal(betas,ATPi,Glc)
    fPDH = find_fPDH(outros,table3,NADHm,Cam)
    JpTCA = (Jredbasal/3)+(0.84*fPDH*dJglytotal)
    return JpTCA

def find_Jpgly(betas,ATPi,Glc):
    dJglytotal = find_dJglytotal(betas,ATPi,Glc)
    Jpgly = 2*dJglytotal
    return Jpgly 

def find_Jhyd(table3,ATPi,dJhyd):
    khyd = table3['khyd']
    Jhyd = (khyd*ATPi)+dJhyd
    return Jhyd

def find_dJhydss(table3,Glc):
    dJhydmax = table3['dJhydmax']
    kGlc = table3['kGlc']
    nhyd = table3['nhyd']
    dJhydss = dJhydmax/(1+(np.power((kGlc/Glc),nhyd)))
    return dJhydss
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MODELOS ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Antes da adição de ADP:
def modelo1(ci,t,outros,tabela1,tabela2,tabela3,tabela4,tabela5,betas,table3,inputs,u):
	Atoti = outros['Atoti']
	CCa2m = table3['CCa2m']
	Cmito = table3['Cmito']
	tauhyd = table3['tauhyd']
	Cai = 0.2					
	NADHm = ci[0]
	ADPm = ci[1]
	Cam = ci[2]
	dp = ci[3]
	ADPi = ci[4]
	dJhyd = ci[5]
	ATPi = Atoti-ADPi
	nadhm = (find_Jred(outros,betas,table3,NADHm,ATPi,Cam,u[inputs.get("Glc")]))-(find_Jo(outros,tabela1,tabela2,NADHm,dp))
	adpm = (find_Jant(outros,tabela1,tabela4,ADPm,ADPi,dp))-(find_JpTCA(outros,betas,table3,NADHm,ATPi,Cam,u[inputs.get("Glc")]))-(find_JpF1(outros,tabela1,tabela3,ADPm,dp,u[inputs.get("oligo")]))
	cam = (1/CCa2m)*((find_Juni(outros,tabela1,tabela5,Cai,dp))-(find_JNaCa(outros,tabela1,tabela5,Cam,dp)))
	dpm = (-1/Cmito)*((find_JHF1(outros,tabela1,tabela3,ADPm,dp,u[inputs.get("oligo")]))+(find_JHleak(outros,tabela1,dp,u[inputs.get("fccp")]))+(find_Jant(outros,tabela1,tabela4,ADPm,ADPi,dp))+(2*(find_Juni(outros,tabela1,tabela5,Cai,dp)))+(find_JNaCa(outros,tabela1,tabela5,Cam,dp))-(find_JHres(outros,tabela1,tabela2,NADHm,dp)))
	adpi = (0.09/60000)*((find_Jhyd(table3,ATPi,dJhyd))-(find_Jpgly(betas,ATPi,u[inputs.get("Glc")]))-(find_Jant(outros,tabela1,tabela4,ADPm,ADPi,dp)))
	djhyd = (1/tauhyd)*((find_dJhydss(table3,u[inputs.get("Glc")]))-dJhyd)
	dzdt = [nadhm,adpm,cam,dpm,adpi,djhyd]
	return dzdt
# Depois da adição de ADP:
def modelo2(ci,t,outros,tabela1,tabela2,tabela3,tabela4,tabela5,betas,table3,inputs,u):
    CCa2m = table3['CCa2m']
    Cmito = table3['Cmito']
    Cai = 0.2					
    NADHm = ci[0]
    ADPm = ci[1]
    Cam = ci[2]
    dp = ci[3]
    # Ajuste da razão ATPi/ADPi após a adição de ADP -----------------------------------------------------
    ADPi = 2.0
    ATPi = 8.0
    total = ADPi+ATPi
    # ----------------------------------------------------------------------------------------------------
    nadhm = (find_Jred(outros,betas,table3,NADHm,ATPi,Cam,u[inputs.get("Glc")]))-(find_Jo(outros,tabela1,tabela2,NADHm,dp))
    adpm = (find_Jant2(outros,tabela1,tabela4,ADPm,ADPi,total,dp))-(find_JpTCA(outros,betas,table3,NADHm,ATPi,Cam,u[inputs.get("Glc")]))-(find_JpF1(outros,tabela1,tabela3,ADPm,dp,u[inputs.get("oligo")]))
    cam = (1/CCa2m)*((find_Juni(outros,tabela1,tabela5,Cai,dp))-(find_JNaCa(outros,tabela1,tabela5,Cam,dp)))
    dpm = (-1/Cmito)*((find_JHF1(outros,tabela1,tabela3,ADPm,dp,u[inputs.get("oligo")]))+(find_JHleak(outros,tabela1,dp,u[inputs.get("fccp")]))+(find_Jant2(outros,tabela1,tabela4,ADPm,ADPi,total,dp))+(2*(find_Juni(outros,tabela1,tabela5,Cai,dp)))+(find_JNaCa(outros,tabela1,tabela5,Cam,dp))-(find_JHres(outros,tabela1,tabela2,NADHm,dp)))
    dzdt = [nadhm,adpm,cam,dpm]
    return dzdt
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# TEMPO  
um_min = 60000                                      # Milisegundos em um minuto
min = 1000                                          # Quantidade de pontos por minuto de simulação
q_min = 10                                          # Quantidade de minutos da simulação
t_i = 0.0                                           # Tempo inicial
t_f = um_min*q_min                                  # Tempo final
N = min*q_min                                       # Número total de pontos                                       
t = np.linspace(t_i,t_f,N)                          # Vetor tempo (ms)
t_n = t*(1/um_min)                                  # Vetor tempo (min)
dt = 10000                                          # Intervalo de tempo da rampa

# ENTRADAS --------------------------------------------------------------------------------------------------------------------------------------------------------------
inputs = {"Glc":0,"oligo":1,"fccp":2}
																				
# Parâmetros Glc
Glc_a = 1.4																		# 13/08
Glc_d = 7.0																		# 13/08
t_Glc = 120000                                  # (ms)
inclinacao_Glc = (Glc_d-Glc_a)/dt
constante_Glc = Glc_a-(inclinacao_Glc*t_Glc) 
# Parâmetros ADP
t_ADP = 4                                       # (min)
# Parâmetros oligo
oligo_a = 1.0
oligo_adi = 0.2																	# 13/08			range_oligo = 0 até 0.2 uL 
oligo_d = 1-(4.7*oligo_adi)														# 13/08
t_oligo = 360000                                # (ms)
inclinacao_oligo = (oligo_d-oligo_a)/dt
constante_oligo = oligo_a-(inclinacao_oligo*t_oligo)
# Parâmetros fccp
fccp_a = 1.0
fccp_adi = 1.0																	# 13/08			range_fccp = 0 até 1.0 uL
fccp_d = 1+(6.5*fccp_adi)														# 13/08
t_fccp = 480000                                 # (ms)
inclinacao_fccp = (fccp_d-fccp_a)/dt
constante_fccp = fccp_a-(inclinacao_fccp*t_fccp)

# Inicialmente
u = {};
u[inputs.get("Glc")] = Glc_a
u[inputs.get("oligo")] = oligo_a
u[inputs.get("fccp")] = fccp_a

# Aplicação das entradas
def tscrypt(t, u):

    if(t < t_Glc):
        u[inputs.get("Glc")] = Glc_a
        u[inputs.get("oligo")] = oligo_a
        u[inputs.get("fccp")] = fccp_a

    # Rampa Glc
    if(t >= t_Glc and t < (t_Glc+dt)):
        u[inputs.get("Glc")] = constante_Glc+(inclinacao_Glc*t)
        u[inputs.get("oligo")] = oligo_a
        u[inputs.get("fccp")] = fccp_a

    if(t >= (t_Glc+dt) and t < t_oligo):
        u[inputs.get("Glc")] = Glc_d
        u[inputs.get("oligo")] = oligo_a
        u[inputs.get("fccp")] = fccp_a

    # Rampa oligo
    if(t >= t_oligo and t < (t_oligo+dt)):
        u[inputs.get("Glc")] = Glc_d
        u[inputs.get("oligo")] = constante_oligo+(inclinacao_oligo*t)
        u[inputs.get("fccp")] = fccp_a

    if(t >= (t_oligo+dt) and t < t_fccp):
        u[inputs.get("Glc")] = Glc_d
        u[inputs.get("oligo")] = oligo_d
        u[inputs.get("fccp")] = fccp_a

    # Rampa fccp
    if(t >= t_fccp and t < (t_fccp+dt)):
        u[inputs.get("Glc")] = Glc_d
        u[inputs.get("oligo")] = oligo_d
        u[inputs.get("fccp")] = constante_fccp+(inclinacao_fccp*t)

    if(t >= (t_fccp+dt)):
        u[inputs.get("Glc")] = Glc_d
        u[inputs.get("oligo")] = oligo_d
        u[inputs.get("fccp")] = fccp_d

    return u

# Aplicação de ADP
N_ADP = 4*min
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Condições Iniciais 
NADHm = 0.02
ADPm = 8.2
Cam = 0.004
dp = 150   
ADPi = 0.6
dJhyd = 0.7																		# 13/08											                       
z0 = [NADHm,ADPm,Cam,dp,ADPi,dJhyd]   

# Armazenamento de soluções
NADHm = np.empty_like(t)
ADPm = np.empty_like(t)
Cam = np.empty_like(t)
dp = np.empty_like(t)
ADPi = np.empty_like(t)
dJhyd = np.empty_like(t)
Jo = np.empty_like(t)
JHres = np.empty_like(t)
JHleak = np.empty_like(t)
JpF1 = np.empty_like(t)
JHF1 = np.empty_like(t)
Juni = np.empty_like(t)
JNaCa = np.empty_like(t)
Jant = np.empty_like(t)
Jred = np.empty_like(t)
JpTCA = np.empty_like(t)
Jpgly = np.empty_like(t)
Jhyd = np.empty_like(t)
dJhydss = np.empty_like(t)

NADHm[0] = z0[0]
ADPm[0] = z0[1]
Cam[0] = z0[2]
dp[0] = z0[3]
ADPi[0] = z0[4]
dJhyd[0] = z0[5]
Jo[0] = find_Jo(outros,tabela1,tabela2,NADHm[0],dp[0])
JHres[0] = find_JHres(outros,tabela1,tabela2,NADHm[0],dp[0])
JHleak[0] = find_JHleak(outros,tabela1,dp[0],fccp_a)
JpF1[0] = find_JpF1(outros,tabela1,tabela3,ADPm[0],dp[0],oligo_a)
JHF1[0] = find_JHF1(outros,tabela1,tabela3,ADPm[0],dp[0],oligo_a)
Juni[0] = find_Juni(outros,tabela1,tabela5,0.2,dp[0])
JNaCa[0] = find_JNaCa(outros,tabela1,tabela5,Cam[0],dp[0])
Jant[0] = find_Jant(outros,tabela1,tabela4,ADPm[0],ADPi[0],dp[0])
Jred[0] = find_Jred(outros,betas,table3,NADHm[0],((outros['Atoti'])-ADPi[0]),Cam[0],Glc_a)
JpTCA[0] = find_JpTCA(outros,betas,table3,NADHm[0],((outros['Atoti'])-ADPi[0]),Cam[0],Glc_a)
Jpgly[0] = find_Jpgly(betas,((outros['Atoti'])-ADPi[0]),Glc_a)
Jhyd[0] = find_Jhyd(table3,((outros['Atoti'])-ADPi[0]),dJhyd[0])
dJhydss[0] = find_dJhydss(table3,Glc_a)

# Para os gráficos 
# ---------- Durante toda a corrida
Ca_c = 0.2                                              
# ---------- Após a adição de ADP (segundo laço for)
ADP_c = 2.0
ATP_c = 8.0
tot = ADP_c+ATP_c
d_Jhyd = 0.0

# Solução do sistema de EDO's 
for i in range(1,N_ADP):
	tspan = [t[i-1],t[i]]
	z = odeint(modelo1,z0,tspan,args=(outros,tabela1,tabela2,tabela3,tabela4,tabela5,betas,table3,inputs,u,))
	NADHm[i] = z[1][0]
	ADPm[i] = z[1][1]
	Cam[i] = z[1][2]
	dp[i] = z[1][3]
	ADPi[i] = z[1][4]
	dJhyd[i] = z[1][5]
	Jo[i] = find_Jo(outros,tabela1,tabela2,NADHm[i],dp[i])
	JHres[i] = find_JHres(outros,tabela1,tabela2,NADHm[i],dp[i])
	JHleak[i] = find_JHleak(outros,tabela1,dp[i],u[inputs.get("fccp")])
	JpF1[i] = find_JpF1(outros,tabela1,tabela3,ADPm[i],dp[i],u[inputs.get("oligo")])
	JHF1[i] = find_JHF1(outros,tabela1,tabela3,ADPm[i],dp[i],u[inputs.get("oligo")])
	Juni[i] = find_Juni(outros,tabela1,tabela5,Ca_c,dp[i])
	JNaCa[i] = find_JNaCa(outros,tabela1,tabela5,Cam[i],dp[i])
	Jant[i] = find_Jant(outros,tabela1,tabela4,ADPm[i],ADPi[i],dp[i])
	Jred[i] = find_Jred(outros,betas,table3,NADHm[i],((outros['Atoti'])-ADPi[i]),Cam[i],u[inputs.get("Glc")])
	JpTCA[i] = find_JpTCA(outros,betas,table3,NADHm[i],((outros['Atoti'])-ADPi[i]),Cam[i],u[inputs.get("Glc")])
	Jpgly[i] = find_Jpgly(betas,((outros['Atoti'])-ADPi[i]),u[inputs.get("Glc")])
	Jhyd[i] = find_Jhyd(table3,((outros['Atoti'])-ADPi[i]),dJhyd[i])
	dJhydss[i] = find_dJhydss(table3,u[inputs.get("Glc")])
	z0 = z[1]
	u = tscrypt(t[i],u)

y0 = [NADHm[i],ADPm[i],Cam[i],dp[i]]

for i in range(N_ADP,N):
	tspan = [t[i-1],t[i]]
	y = odeint(modelo2,y0,tspan,args=(outros,tabela1,tabela2,tabela3,tabela4,tabela5,betas,table3,inputs,u,))
	NADHm[i] = y[1][0]
	ADPm[i] = y[1][1]
	Cam[i] = y[1][2]
	dp[i] = y[1][3]
	ADPi[i] = ADP_c
	dJhyd[i] = d_Jhyd
	Jo[i] = find_Jo(outros,tabela1,tabela2,NADHm[i],dp[i])
	JHres[i] = find_JHres(outros,tabela1,tabela2,NADHm[i],dp[i])
	JHleak[i] = find_JHleak(outros,tabela1,dp[i],u[inputs.get("fccp")])
	JpF1[i] = find_JpF1(outros,tabela1,tabela3,ADPm[i],dp[i],u[inputs.get("oligo")])
	JHF1[i] = find_JHF1(outros,tabela1,tabela3,ADPm[i],dp[i],u[inputs.get("oligo")])
	Juni[i] = find_Juni(outros,tabela1,tabela5,Ca_c,dp[i])
	JNaCa[i] = find_JNaCa(outros,tabela1,tabela5,Cam[i],dp[i])
	Jant[i] = find_Jant2(outros,tabela1,tabela4,ADPm[i],ADPi[i],tot,dp[i])
	Jred[i] = find_Jred(outros,betas,table3,NADHm[i],ATP_c,Cam[i],u[inputs.get("Glc")])
	JpTCA[i] = find_JpTCA(outros,betas,table3,NADHm[i],ATP_c,Cam[i],u[inputs.get("Glc")])
	y0 = y[1]
	u = tscrypt(t[i],u)

# Gráficos
"""
Jo_n = Jo*(1250/60000)
plt.plot(t_n,Jo_n,'r',linewidth=3.0)
plt.axis([0.1,10.1,0.38,0.48])
plt.xlabel('tempo (min)',fontsize=18)
plt.ylabel('Jo (uM/ms)',fontsize=18)
plt.tick_params(labelsize=12)
plt.show()
"""
plt.figure(1)
plt.plot(t_n[10:],Jo[10:],'crimson',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jo (nmol/min/mg-protein)',fontsize=15)
plt.savefig('Jo.png', format='png')

plt.figure(2)
plt.plot(t_n[10:],NADHm[10:],'goldenrod',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('NADHm (nmol/mg-protein)',fontsize=15)
plt.savefig('NADHm.png', format='png')

plt.figure(3)
plt.plot(t_n[10:],ADPm[10:],'c',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('ADPm (nmol/mg-protein)',fontsize=15)
plt.savefig('ADPm.png', format='png')

# Ajuste
Cam_n = Cam*1000
plt.figure(4)
plt.plot(t_n[10:],Cam_n[10:],'peru',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Cam (pmol/mg-protein)',fontsize=15)
plt.savefig('Cam.png', format='png')

plt.figure(5)
plt.plot(t_n[10:],dp[10:],'lime',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('dp (mV)',fontsize=15)
plt.savefig('dp.png', format='png')

plt.figure(6)
plt.plot(t_n[10:],ADPi[10:],'violet',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('ADPi (mM)',fontsize=15)
plt.savefig('ADPi.png', format='png')

plt.figure(7)
plt.plot(t_n[10:],dJhyd[10:],'coral',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('dJhyd (nmol/(mg-protein*ms))',fontsize=15)
plt.savefig('dJhyd.png', format='png')

plt.figure(8)
plt.plot(t_n[10:],JHres[10:],'blueviolet',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JHres (nmol/min/mg-protein)',fontsize=15)
plt.savefig('JHres.png', format='png')

plt.figure(9)
plt.plot(t_n[10:],JHleak[10:],'lightcoral',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JHleak (nmol/min/mg-protein)',fontsize=15)
plt.savefig('JHleak.png', format='png')

plt.figure(10)
plt.plot(t_n[10:],JpF1[10:],'teal',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JpF1 (nmol/min/mg-protein)',fontsize=15)
plt.savefig('JpF1.png', format='png')

plt.figure(11)
plt.plot(t_n[10:],JHF1[10:],'steelblue',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JHF1 (nmol/min/mg-protein)',fontsize=15)
plt.savefig('JHF1.png', format='png')

plt.figure(12)
plt.plot(t_n[10:],Juni[10:],'darkorange',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Juni (nmol/min/mg-protein)',fontsize=15)
plt.savefig('Juni.png', format='png')

plt.figure(13)
plt.plot(t_n[10:],JNaCa[10:],'gold',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JNaCa (nmol/min/mg-protein)',fontsize=15)
plt.savefig('JNaCa.png', format='png')

plt.figure(14)
plt.plot(t_n[10:],Jant[10:],'mediumorchid',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jant (nmol/min/mg-protein)',fontsize=15)
plt.savefig('Jant.png', format='png')

plt.figure(15)
plt.plot(t_n[10:],Jred[10:],'chocolate',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jred (nmol/min/mg-protein)',fontsize=15)
plt.savefig('Jred.png', format='png')

plt.figure(16)
plt.plot(t_n[10:],JpTCA[10:],'rebeccapurple',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('JpTCA (nmol/min/mg-protein)',fontsize=15)
plt.savefig('JpTCA.png', format='png')

plt.figure(17)
plt.plot(t_n[10:],Jpgly[10:],'darkgoldenrod',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jpgly (nmol/min/mg-protein)',fontsize=15)
plt.savefig('Jpgly.png', format='png')

plt.figure(18)
plt.plot(t_n[10:],Jhyd[10:],'darkslateblue',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('Jhyd',fontsize=15)
plt.savefig('Jhyd.png', format='png')

plt.figure(19)
plt.plot(t_n[10:],dJhydss[10:],'mediumseagreen',linewidth=2.5)
plt.xlabel('time (min)',fontsize=15)
plt.ylabel('dJhydss',fontsize=15)
plt.savefig('dJhydss.png', format='png')