import xml.etree.ElementTree as ET
import pandas as pd
import os
import re

Path = "./Data+location/Data/England/"
df_locate = pd.read_csv("./Data+location/dblocks-location-address.csv", sep=";")
for i in os.walk(Path):
    fileName = i[2]

fileName_nosuffix = []
fileName_all = []
for j in range(len(fileName)):
    if fileName[j] == ".DS_Store":
        print()
    else:
        fileName_all.append(fileName[j])
        fileName_nosuffix.append(fileName[j][0:-4])
#fileName_nosuffix is a list of filename without suffix ".xml" in the folder
data_list=[]
for i in range(len(fileName)):
    list_8 = ['','','','','','','','','']
    columns = ['fileName', 'dblock_Id', 'tile', 'year', 'content', 'Town', 'county', 'region', 'country']
    list_8[0] = fileName_nosuffix[i]
    filePath = Path + fileName[i]
    text = open(filePath, encoding='utf-8').read()
    # Replace illegal characters in xml with spaces
    text = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+", u"", text)
    root = ET.fromstring(text)
    for k in root.iter("dblock_id"):
        list_8[1] = str(k.text)
    for k in root.iter("title"):
        list_8[2] = str(k.text)
    for k in root.iter("year"):
        list_8[3] = str(k.text)
    for k in root.iter("content"):
        list_8[4] = str(k.text).replace('\n',"").replace("  ","")
    # list_8[4:7] = list(df_locate.loc[list_8[1],['Town','county','region','country']])
    # list_8[4:8] = list(df_locate.loc[df_locate['Dblock']==list_8[1]])[1:5]
    search_loc = df_locate.loc[df_locate['Dblock']==list_8[1],['Town','county','region','country']]
    list_8[5:9] = search_loc.values.tolist()[0]
    data_list.append(list_8)
df_result = pd.DataFrame(data_list, columns=columns)
df_result.to_csv('England.csv')
print(df_result)