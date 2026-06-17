import zipfile 
z=zipfile.ZipFile('models/Multi-Cas_1IN_9Hydra_Divide_Testset.keras','r') 
print(z.namelist()) 
