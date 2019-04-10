import os
import numpy as np
import pickle
import csv

def main():

    distDiff = 200
    f = open('/cs/cbio/hofit/Data/Illumina_450k.sort.txt', 'r')
    f2 = open('/cs/cbio/hofit/Data/sorted_hg19.CpG.bed', 'r')
    line = f.readline()
    line2 = f2.readline()
    line = line.split()
    line2 = line2.split()
    cgName = line[0]
    chr = line[1]
    location = int(line[2])
    cgClusters = []
    clustersNames = ['cg site']
    while(line):
        currCluster = []
        clustersNames.append(cgName)
        while(line2):
            if(chr==line2[0] and abs(location-int(line2[1]))<=distDiff):
                currCluster.append(line2[3])
                line2 = f2.readline()
                line2 = line2.split()
            elif(location > int(line2[1])):
                line2 = f2.readline()
                line2 = line2.split()
            else:
                break
        cgClusters.append(currCluster)
        line = f.readline()
        while(line):
            line = line.split()
            if(line[2] == 'MAPINFO'):
                line = ''
                break
            if(chr != line[1] or abs(int(line[2])-location)>distDiff):
                cgName = line[0]
                chr = line[1]
                location = int(line[2])
                break
            else:
                clustersNames.append(line[0])
                cgClusters.append([])
                line = f.readline()
    f.close()
    f2.close()

    allPatientsData = [clustersNames]
    coverThresh = 5
    for filename in os.listdir('/cs/cbio/netanel/data/guo_data/RRBS'):
        if filename.endswith(".beta"):
            sample = np.fromfile('/cs/cbio/netanel/data/guo_data/RRBS/'+filename, dtype=np.uint8).reshape((-1, 2))
            sumCpgValue = 0;
            sumCpgCovers = 0;
            avgCpg = 0;
            avgArrForTCGA = [filename[21:-5]] #name of the patient
            #coverArrForTCGA = []
            for i in range(len(cgClusters)):
                if(len(cgClusters[i]) != 0):
                    for clusterName in cgClusters[i]:
                        clusterNum = int(clusterName[3:])
                        sumCpgValue += sample[clusterNum-1][0]
                        sumCpgCovers += sample[clusterNum-1][1]
                    if(sumCpgCovers != 0):
                        avgCpg = int((sumCpgValue/sumCpgCovers)*1000)
                    else:
                        avgCpg = 'NaN'

                if(sumCpgCovers >= coverThresh):
                    avgArrForTCGA.append(avgCpg)
                else:
                    avgArrForTCGA.append('NaN')
                    
                sumCpgValue = 0;
                sumCpgCovers = 0;
                avgCpg = 0;
            allPatientsData.append(avgArrForTCGA)
                #coverArrForTCGA.append(sumCpgCovers)
        else:
            continue

    allPatientsData = np.array(allPatientsData).transpose()

    with open("bloodPatientData.csv", "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(allPatientsData)
        output.close()

    pickle.dump(allPatientsData, open("bloodPatientData.pickle", "wb"), protocol=4)
    #np.savetxt("bloodPatientData.csv", allPatientsData)




if __name__ == '__main__':
    main()