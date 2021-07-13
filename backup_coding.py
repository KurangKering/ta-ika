import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from numpy import unravel_index
import itertools

class FuzzyIndex:
    
    @staticmethod
    def get_index(x):
        indexes = []
        for i in range(len(x)):
            matching = np.argwhere(x[i] > 0).flatten()
            indexes.append(matching.tolist())
        return indexes
    
    

class MF_Age(FuzzyIndex):
    
    __bobot = [0.3, 0.5, 0.7, 0.8]
    __himpunan = ['muda', 'agak tua', 'tua', 'sangat tua']
    
    @staticmethod
    def get_himpunan(index):
        return MF_Age.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Age.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        _muda = fuzz.trapmf(x, [0, 0, 33, 38])
        _agak_tua = fuzz.trapmf(x, [33, 38, 40, 45])
        _tua = fuzz.trapmf(x, [40, 45, 52, 58])

        _replaced_x = np.array(list(map(lambda y : y if y <= 58 else 58, x)))

        _sangat_tua = fuzz.trapmf(_replaced_x, [52, 58, 58, 58])
        
        _merged = np.vstack((_muda, _agak_tua, _tua, _sangat_tua))
        transposed = _merged.T
        return transposed
    

        



class MF_Trestbps(FuzzyIndex):
    
    __bobot = [0.3, 0.5, 0.7, 0.9]
    __himpunan = ['rendah', 'sedang', 'tinggi', 'sangat tinggi']
    
    @staticmethod
    def get_himpunan(index):
        return MF_Trestbps.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Trestbps.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        _rendah = fuzz.trapmf(x, [0, 0, 127, 134])
        _sedang = fuzz.trapmf(x, [127, 134, 142, 153])
        _tinggi = fuzz.trapmf(x, [142, 153, 154, 172])

        _replaced_x = np.array(list(map(lambda y : y if y <= 172 else 172, x)))

        _sangat_tinggi = fuzz.trapmf(_replaced_x, [154, 172, 172, 172])
        
        _merged = np.vstack((_rendah, _sedang, _tinggi, _sangat_tinggi))
        transposed = _merged.T
        return transposed


class MF_Chol(FuzzyIndex):
    
    __bobot = [0.5, 0.7, 0.8, 0.9]
    __himpunan = ['rendah', 'sedang', 'tinggi', 'sangat tinggi']
    
    @staticmethod
    def get_himpunan(index):
        return MF_Chol.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Chol.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        _rendah = fuzz.trapmf(x, [0, 0, 188, 197])
        _sedang = fuzz.trapmf(x, [188, 197, 217, 250])
        _tinggi = fuzz.trapmf(x, [217, 250, 281, 307])

        _replaced_x = np.array(list(map(lambda y : y if y <= 307 else 307, x)))

        _sangat_tinggi = fuzz.trapmf(_replaced_x, [281, 307, 307, 307])
        
        _merged = np.vstack((_rendah, _sedang, _tinggi, _sangat_tinggi))
        transposed = _merged.T
        return transposed
        


class MF_Thalach(FuzzyIndex):
    
    __bobot = [0.3, 0.5, 0.7]
    __himpunan = ['rendah', 'sedang', 'tinggi']
    
    @staticmethod
    def get_himpunan(index):
        return MF_Thalach.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Thalach.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        _rendah = fuzz.trapmf(x, [0, 0, 111, 141])
        _sedang = fuzz.trapmf(x, [111, 141, 152, 194])

        _replaced_x = np.array(list(map(lambda y : y if y <= 194 else 194, x)))
        _tinggi = fuzz.trapmf(_replaced_x, [152, 194, 194, 194])

        _merged = np.vstack((_rendah, _sedang, _tinggi))
        transposed = _merged.T
        return transposed
        


class MF_Oldpeak(FuzzyIndex):
    
    __bobot = [0.5, 0.5, 0.5]
    __himpunan = ['rendah', 'beresiko', 'buruk']
    
    @staticmethod
    def get_himpunan(index):
        return MF_Oldpeak.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Oldpeak.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        _rendah = fuzz.trapmf(x, [0, 0, 1.5, 2])
        _sedang = fuzz.trapmf(x, [1.5, 2, 2.55, 4.2])

        _replaced_x = np.array(list(map(lambda y : y if y <= 4.2 else 4.2, x)))
        _tinggi = fuzz.trapmf(_replaced_x, [2.55, 4.2, 4.2, 4.2])

        _merged = np.vstack((_rendah, _sedang, _tinggi))
        transposed = _merged.T
        return transposed
        


class MF_Sex:
    
    __bobot = [0.4, 0.5]
    __himpunan = ['perempuan', 'laki-laki']

    @staticmethod
    def get_himpunan(index):
        return MF_Sex.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Sex.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        if any(i > 1 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 1')

        return x[:, None]
    
    @staticmethod
    def get_index(x):
        if any(i > 1 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 1')
            
        indexes = []
        for i in range(len(x)):
            
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes
    


class MF_Cp:
    
    __bobot = [0.7, 0.7, 0.7, 0.7]
    __himpunan = ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']

    @staticmethod
    def get_himpunan(index):
        return MF_Cp.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Cp.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        if any(i > 4 for i in x) or any(i < 1 for i in x):
            raise ValueError('nilai berada diantara 1 dan 4')
         
        new_x = np.array(list(map(lambda y : y-1 , x)))

        return new_x[:, None]
    
    @staticmethod
    def get_index(x):
        
        if any(i > 3 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 3')
            
        indexes = []
        for i in range(len(x)):
            
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes
    


class MF_Fbs:
    
    __bobot = [0.4, 0.8]
    __himpunan = ['tidak', 'ya']

    @staticmethod
    def get_himpunan(index):
        return MF_Fbs.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Fbs.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        new_x = np.array(list(map(lambda y : 0 if y <= 120 else 1, x)))
        return new_x[:, None]
        
    @staticmethod
    def get_index(x):
        if any(i > 1 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 1')
            
        indexes = []
        for i in range(len(x)):
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes
    


class MF_Restach:
    
    __bobot = [0.2, 0.9, 0.9]
    __himpunan = ['normal', 'ST-T wave abnormality', 'hypertrophy']

    @staticmethod
    def get_himpunan(index):
        return MF_Restach.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Restach.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        if any(i > 2 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 2')

        return x[:, None]
    
         
    @staticmethod
    def get_index(x):
        if any(i > 2 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 2')
            
        indexes = []
        for i in range(len(x)):
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes


class MF_Exang:
    
    __bobot = [0.5, 0.5]
    __himpunan = ['nyeri', 'tidak nyeri']

    @staticmethod
    def get_himpunan(index):
        return MF_Exang.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Exang.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        if any(i > 1 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 1')
        
        new_x = np.array(list(map(lambda y: 1 if y == 0 else 0, x)))
        return new_x[:, None]
          
    @staticmethod
    def get_index(x):
        if any(i > 1 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 1')
            
        indexes = []
        for i in range(len(x)):
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes
    


class MF_Slope:
    
    __bobot = [0.5, 0.5,0.5]
    __himpunan = ['upsloping', 'flat', 'downsloping']

    @staticmethod
    def get_himpunan(index):
        return MF_Slope.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Slope.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        if any(i > 3 for i in x) or any(i < 1 for i in x):
            raise ValueError('nilai berada diantara 1 dan 3')
        
        new_x = np.array(list(map(lambda y: y - 1, x)))
        return new_x[:, None]

    @staticmethod
    def get_index(x):
        if any(i > 2 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 2')
            
        indexes = []
        for i in range(len(x)):
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes


class MF_Ca:
    
    __bobot = [0.5, 0.5, 0.5, 0.5]
    __himpunan = ['ca1', 'ca2', 'ca3', 'ca4']

    @staticmethod
    def get_himpunan(index):
        return MF_Ca.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Ca.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        if any(i > 3 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 3')

        return x[:, None]
    
    
    @staticmethod
    def get_index(x):
        if any(i > 3 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 3')
            
        indexes = []
        for i in range(len(x)):
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes



class MF_Thal:
    
    __himpunan = ['normal', 'fixed defect', 'reversible defect']
    __bobot = [0.3, 0.7, 0.8]

    @staticmethod
    def get_himpunan(index):
        return MF_Thal.__himpunan[index]
    
    @staticmethod
    def get_bobot(index):
        return MF_Thal.__bobot[index]
    
    @staticmethod
    def single_calculate(x):
        new_x = np.array(list(map(lambda y : 0 if y < 5.5 else 2 if y > 6.2 else 1, x)))
        return new_x[:, None]

        return x[:, None]

    @staticmethod
    def get_index(x):
        if any(i > 2 for i in x) or any(i < 0 for i in x):
            raise ValueError('nilai berada diantara 0 dan 2')
            
        indexes = []
        for i in range(len(x)):
            matching = x[i]
            indexes.append(matching.tolist())
        return indexes


x = np.array([63, 145, 233, 150, 2.3, 1, 3, 1, 0, 0, 1, 0, 0])
lib = [MF_Age, MF_Trestbps, MF_Chol, MF_Thalach, MF_Oldpeak, MF_Sex, MF_Cp, MF_Fbs, MF_Restach, MF_Exang, MF_Slope, MF_Ca, MF_Thal]

def getNilaiFuzzy(arr):
    return [lib[x].single_calculate(arr[x]) for x in range(len(arr))]

getNilaiFuzzy(x)

for i in range(len(aturan_tiap_data)):
    for j in range(len(aturan_tiap_data[i])):
        temp = []
        for k in range(len(aturan_tiap_data[i][j])):
            temp.append(lib[k].get_himpunan(int(aturan_tiap_data[i][j][k])))
            
        print(' '.join(temp))