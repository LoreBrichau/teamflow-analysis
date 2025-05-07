import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import bootstrap


def gini(array):
    array = np.array(array)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001  # Voorkom deling door nul
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

proportie ={'team1' : [0.301528294,0.266831888,0.269723255,0.161916563],
"team2" : [0.26625239,
0.20793499,
0.184034417,
0.341778203],
"team3" : [0.265386787,
0.114059853,
0.304347826,
0.316205534],
"team4" : [0.086712683,
0.493528904,
0.283433995,
0.136324418],
"team5" :[0.313878676,
0.382352941,
0.175551471,
0.128216912],
"team6" : [0.234710744,
0.195867769,
0.387603306,
0.181818182],
"team7" : [0.37690898,
0.173793525,
0.219303604,
0.229993891],
"team8" : [0.251036562,
0.373539389,
0.145872597,
0.229551451],
"team9" : [0.161103048,
0.208272859,
0.33490566,
0.295718433],
"team10" : [0.314363817,
0.436878728,
0.120775348,
0.127982107],
"team11" : [0.304543185,
0.217174239,
0.141537693,
0.336744883],
"team13" : [0.076164258,
0.559761413,
0.225051617,
0.139022712],
"team14": [0.230961015,
0.417497733,
0.200362647,
0.151178604],
"team15" : [0.294117647,
0.321350763,
0.188453159,
0.196078431]}

gini_lijst = []

for team_key in proportie.keys():
    gini_lijst.append(gini(proportie[team_key]))
print(gini_lijst)



scorelist = []
teamlist = []



df = pd.read_excel('Teamflowmonitor.xlsx', usecols='NZ:OM')

for key in df.keys():
    for i in range(0,60):
        scorelist.append(df[key][i])

print(len(scorelist))

# for i in range(1,16):
#         teamlist.append(i)
#         teamlist.append(i)
#         teamlist.append(i)
#         teamlist.append(i)


# data = pd.DataFrame({'team': teamlist,
#                   'score': scorelist[0]})
# data['raters'] = [1,2,3,4]*15
# icc = pg.intraclass_corr(data=data, targets='team', raters='raters', ratings='score')
# print(icc)
teamflow = dict()
teamflow_gem = dict()
team = 0
key = 0
keys = df.keys()

for i in range(1,14*15+1):
    if team < 15: team+=1
    else: team = 1

    if (i-1)%15 == 0 and i != 1 and key <13: key+=1
    onderdeel =keys[key]
    teamflow[str(team),onderdeel] = scorelist[(i-1)*4:i*4]
for key in teamflow.keys():
   print(f' {key}:{gini(teamflow[key])}')



for teamflow_key in teamflow.keys():
    teamflow_gem[teamflow_key] = sum(teamflow[teamflow_key])/4
print(teamflow_gem)
print(teamflow)
print(keys)




df2 = pd.read_excel('Teamobservaties.Output.VOSAIC (1).xlsx',sheet_name='Rangschikking', usecols='A:M')
observaties = dict()
df2_keys = df2.keys()
j=3




for i in range(0, 14):
    observaties[df2[df2_keys[1]][i] ] = [df2[df2_keys[1 + 1]][i]]
while j <12:
    for i in range(0,14):
        observaties[df2[df2_keys[j]][i]] = observaties[df2[df2_keys[j]][i]] + [df2[df2_keys[j+1]][i]]
    j+=2
obs_crit = ['wij_ik','stilte','vragen','spreekwiss','actpiek' ,'onderbr']
print(observaties)


wij_ik = [observaties['Team1'][0],observaties['Team2'][0],observaties['Team3'][0],observaties['Team4'][0],observaties['Team5'][0],observaties['Team6'][0],
observaties['Team7'][0],observaties['Team8'][0],observaties['Team9'][0],observaties['Team10'][0], observaties['Team11'][0],observaties['Team13'][0],observaties['Team14'][0],observaties['Team15'][0]]
stilte = [observaties['Team1'][1],observaties['Team2'][1],observaties['Team3'][1],observaties['Team4'][1],observaties['Team5'][1],observaties['Team6'][1],
observaties['Team7'][1],observaties['Team8'][1],observaties['Team9'][1],observaties['Team10'][1], observaties['Team11'][1],observaties['Team13'][1],observaties['Team14'][1],observaties['Team15'][1]]
vragen = [observaties['Team1'][2],observaties['Team2'][2],observaties['Team3'][2],observaties['Team4'][2],observaties['Team5'][2],observaties['Team6'][2],
observaties['Team7'][2],observaties['Team8'][2],observaties['Team9'][2],observaties['Team10'][2], observaties['Team11'][2],observaties['Team13'][2],observaties['Team14'][2],observaties['Team15'][2]]
spreekw = [observaties['Team1'][3],observaties['Team2'][3],observaties['Team3'][3],observaties['Team4'][3],observaties['Team5'][3],observaties['Team6'][3],
observaties['Team7'][3],observaties['Team8'][3],observaties['Team9'][3],observaties['Team10'][3], observaties['Team11'][3],observaties['Team13'][3],observaties['Team14'][3],observaties['Team15'][3]]
actpiek = [observaties['Team1'][4],observaties['Team2'][4],observaties['Team3'][4],observaties['Team4'][4],observaties['Team5'][4],observaties['Team6'][4],
observaties['Team7'][4],observaties['Team8'][4],observaties['Team9'][4],observaties['Team10'][4], observaties['Team11'][4],observaties['Team13'][4],observaties['Team14'][4],observaties['Team15'][4]]
onderbr = [observaties['Team1'][5],observaties['Team2'][5],observaties['Team3'][5],observaties['Team4'][5],observaties['Team5'][5],observaties['Team6'][5],
observaties['Team7'][5],observaties['Team8'][5],observaties['Team9'][5],observaties['Team10'][5], observaties['Team11'][5],observaties['Team13'][5],observaties['Team14'][5],observaties['Team15'][5]]
proportie = gini_lijst

tot_observaties = [wij_ik,stilte,vragen,actpiek,proportie,spreekw,onderbr]
obs_names = ['wij_ik','stilte','vragen','actpiek','proportie gini','spreekw','onderbr']
teamperformantie= [12.1,6.7,5.7,7.6,8.1,8.5,10.7,10.2,8.6,9.6,16.4,11.8,15.2,6.2]
POST_CollectieveAmbitie=[]
POST_GezamenlijkDoel=[]
POST_PersoonlijkeDoelen = []
POST_BundelingVanKrachten = []
POST_OpenCommunicatie = []
POST_VeiligKlimaat = []
POST_WederzijdsCommitment = []
POST_GevoelVanEenheid = []
for key in teamflow_gem.keys():
    if 'POST_CollectieveAmbitie' in key:
        POST_CollectieveAmbitie.append(teamflow_gem[key])
    if 'POST_GezamenlijkDoel' in key:
        POST_GezamenlijkDoel.append(teamflow_gem[key])
    if 'POST_PersoonlijkeDoelen' in key:
        POST_PersoonlijkeDoelen.append(teamflow_gem[key])
    if 'POST_BundelingVanKrachten' in key:
        POST_BundelingVanKrachten.append(teamflow_gem[key])
    if 'POST_VeiligKlimaat' in key:
        POST_VeiligKlimaat.append(teamflow_gem[key])
    if 'POST_OpenCommunicatie' in key:
        POST_OpenCommunicatie.append(teamflow_gem[key])
    if 'POST_WederzijdsCommitment' in key:
        POST_WederzijdsCommitment.append(teamflow_gem[key])
    if 'POST_GevoelVanEenheid' in key:
        POST_GevoelVanEenheid.append(teamflow_gem[key])
POST_CollectieveAmbitie.pop(11)
POST_GezamenlijkDoel.pop(11)
POST_PersoonlijkeDoelen.pop(11)
POST_BundelingVanKrachten.pop(11)
POST_OpenCommunicatie.pop(11)
POST_VeiligKlimaat.pop(11)
POST_WederzijdsCommitment.pop(11)
POST_GevoelVanEenheid.pop(11)
tot_teamflow = [POST_CollectieveAmbitie,POST_GezamenlijkDoel,POST_PersoonlijkeDoelen,POST_WederzijdsCommitment,POST_OpenCommunicatie,POST_VeiligKlimaat,POST_BundelingVanKrachten]
teamflow_names = ['POST_CollectieveAmbitie','POST_GezamenlijkDoel','POST_PersoonlijkeDoelen','POST_WederzijdsCommitment','POST_OpenCommunicatie','POST_VeiligKlimaat','POST_BundelingVanKrachten']





# Bootstrap function


def bootstrap_spearman(data_1, data_2, n_bootstrap=2000, confidence_level=0.90):
    np.random.seed(42)  # For reproducibility
    bootstrapped_corrs = []

    n = len(data_1)
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.choice(range(n), size=n, replace=True)
        sample_1 = data_1[indices]
        sample_2 = data_2[indices]

        # Calculate Spearman correlation for the resample
        corr, pwaarde = spearmanr(sample_1, sample_2)
        bootstrapped_corrs.append(corr)

    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrapped_corrs, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_corrs, (1 + confidence_level) / 2 * 100)

    return np.mean(bootstrapped_corrs), (lower_bound, upper_bound),pwaarde

tot_observaties.append(teamperformantie)
obs_names.append('teamperformantie')
# Apply bootstrapping
obs = 0
flow = 0
for data1 in tot_observaties:
    obs+=1
    flow = 0
    for data2 in tot_teamflow:
        flow+=1
        if (obs ==1 and (flow ==1 or flow ==2 or flow==3 or flow ==4)) or ((obs==2 or obs ==3) and flow==5 ) or (obs==4 and flow ==6) or ((obs == 7 or obs==5 or obs==6)and flow==7) or obs ==8:
            print(obs_names[obs-1] , teamflow_names[flow-1])
            mean_corr, conf_interval, pwaarde = bootstrap_spearman(np.array(data1), np.array(data2))
            print(f"Bootstrapped Spearman Correlation: {mean_corr}")
            print(f"90% Confidence Interval: {conf_interval}")
            print(f'p waarde: {pwaarde}')
            print()



















