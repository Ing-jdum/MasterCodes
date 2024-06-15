# jugar_tenis.py
# Ejemplo visto en clase (ver diapositivas)

# atributos=[('Cielo',['Soleado','Nublado','Lluvia']),
#            ('Temperatura',['Alta','Baja','Suave']),
#            ('Humedad',['Alta','Normal']),
#            ('Viento',['Débil','Fuerte'])]

# atributo_clasificación='Jugar Tenis'
# clases=['si','no']

import numpy as np


X_tenis=np.array([['Soleado' , 'Alta'        , 'Alta'    , 'Débil'], 
                  ['Soleado' , 'Alta'        , 'Alta'    , 'Fuerte'], 
                  ['Nublado' , 'Alta'        , 'Alta'    , 'Débil'],  
                  ['Lluvia'  , 'Suave'       , 'Alta'    , 'Débil'],  
                  ['Lluvia'  , 'Baja'        , 'Normal'  , 'Débil' ], 
                  ['Lluvia'  , 'Baja'        , 'Normal'  , 'Fuerte'], 
                  ['Nublado' , 'Baja'        , 'Normal'  , 'Fuerte'], 
                  ['Soleado' , 'Suave'       , 'Alta'    , 'Débil'],  
                  ['Soleado' , 'Baja'        , 'Normal'  , 'Débil'],  
                  ['Lluvia'  , 'Suave'       , 'Normal'  , 'Débil'],  
                  ['Soleado' , 'Suave'       , 'Normal'  , 'Fuerte'], 
                  ['Nublado' , 'Suave'       , 'Alta'    , 'Fuerte'], 
                  ['Nublado' , 'Alta'        , 'Normal'  , 'Débil'],  
                  ['Lluvia'  , 'Suave'       , 'Alta'    , 'Fuerte']])

y_tenis=np.array(['no',
                  'no',      
                  'si',
                  'si',
                  'si',
                  'no',
                  'si',
                  'no',
                  'si',
                  'si',
                  'si',
                  'si',
                  'si',
                  'no'])
