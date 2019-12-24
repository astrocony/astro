from scipy import ndimage
from scipy import misc 
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# Pregunta 1:

# Get galaxy image

image_file_galaxia='galaxia.jpg'

# Read in RGB

f_color_galaxia=misc.imread(image_file_galaxia, mode='RGB')

# Show image

plt.title('Imagen original',fontsize=14)
plt.imshow(f_color_galaxia)
plt.show()

#  Three colors

galaxia_rojo=f_color_galaxia[:,:,0]
galaxia_verde=f_color_galaxia[:,:,1]
galaxia_azul=f_color_galaxia[:,:,2]

print('arreglo galaxia azul',galaxia_azul[0:20,0:20])

plt.title('Capa azul de imagen',fontsize=14)
plt.imshow(galaxia_azul)
plt.colorbar()
plt.show()

# Pregunta 2:

#functions

#exp

#CONDICION NUMEROS NEGATIVOS


#galaxia_azul=np.where(galaxia_azul>=6,5,galaxia_azul)

galaxia_azul=np.where(galaxia_azul==0,1,galaxia_azul)

galaxia_azul_log10=np.log10(galaxia_azul)
galaxia_azul_log10=galaxia_azul_log10.astype(int)


print('arreglo logaritmo base 10',galaxia_azul_log10[0:20,0:20])

plt.title('Imagen con logaritmo base 10',fontsize=14)
plt.imshow(galaxia_azul_log10)
plt.colorbar()
plt.show()

#sqrt


galaxia_azul_sqrt=np.sqrt(galaxia_azul)
galaxia_azul_sqrt=galaxia_azul_sqrt.astype(int)

print('arreglo de raiz cuadrada',galaxia_azul_sqrt[0:20,0:20])

plt.title('Imagen con raiz cuadrada',fontsize=14)
plt.imshow(galaxia_azul_sqrt)
plt.colorbar()
plt.show()

#log

galaxia_azul_log=np.log(galaxia_azul)
galaxia_azul_log=galaxia_azul_log.astype(int)

print('arreglo de logaritmo natural',galaxia_azul_log[0:20,0:20])

plt.title('Imagen con logaritmo natural',fontsize=14)
plt.imshow(galaxia_azul_log)
plt.colorbar()
plt.show()

# imagen para informe 

imagen_funciones=[galaxia_azul_log10,galaxia_azul_sqrt,galaxia_azul_log]

for i,f in enumerate(imagen_funciones):
    titulos=['logaritmo 10','raíz cuadrada','logaritmo natural']
    plt.subplot(1,3,i+1)
    plt.title(titulos[i],fontsize='8')
    plt.imshow(f)   
plt.show()


#grafico de pixeles

#pixeles_1=[galaxia_azul[324:327,228:231],galaxia_azul_exp[324:327,228:231],galaxia_azul_sqrt[324:327,228:231],galaxia_azul_log[324:327,228:231]]

#for i,f in enumerate(pixeles_1):
#    plt.subplot(2,2,i+1)
#   plt.imshow(f)  
#   plt.colorbar()
#plt.show()

# Pregunta 3:

# filtro Gaussiano 

filter_galaxia_azul_gau=ndimage.gaussian_filter(galaxia_azul,sigma=3)

print('filtro gaussiano es:',filter_galaxia_azul_gau)

plt.title('Filtro Gaussiano',fontsize=14)
plt.imshow(filter_galaxia_azul_gau)
plt.colorbar()
plt.show()

#filtro Laplaciano

laplacian_kernel=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])*(1/8)

filter_galaxia_azul_lap=ndimage.convolve(galaxia_azul,laplacian_kernel)

print('filtro laplaciano:',filter_galaxia_azul_lap)

plt.title('filtro laplaciano',fontsize=14)
plt.imshow(filter_galaxia_azul_lap)
plt.colorbar()
plt.show()

#filtro boxcar

boxcar_kernel=np.ones((3,3))*(1/9)

filter_galaxia_azul_box=ndimage.convolve(galaxia_azul,boxcar_kernel)

print('filtro boxcar:',filter_galaxia_azul_box)

plt.title('filtro boxcar',fontsize=14)
plt.imshow(filter_galaxia_azul_box)
plt.colorbar()
plt.show()

#imagen para informe 2

imagen_filtros=[filter_galaxia_azul_gau,filter_galaxia_azul_lap,filter_galaxia_azul_box]

for i,f in enumerate(imagen_filtros):
    titulo=['filtro gaussiano','filtro laplaciano','filtro boxcar']
    plt.subplot(1,3,i+1)
    plt.title(titulo[i])
    plt.imshow(f)
plt.show()

# Pregunta 4:

#forma uno


kernel_identidad_5=np.zeros((5,5))
kernel_identidad_5[2,2]=1       #matriz identidad 5X5

kernel_gau_5=ndimage.gaussian_filter(kernel_identidad_5,sigma=1)  #matriz gaussiana

F=2*(kernel_identidad_5)-kernel_gau_5

sharpened1=ndimage.convolve(galaxia_azul,F)

print('filtro "2I-GAUSSIANO:',sharpened1)

plt.title('Sharpened 1',fontsize=14)
plt.imshow(sharpened1)
plt.colorbar()
plt.show()

#forma dos

sharpened2=ndimage.convolve(galaxia_azul,laplacian_kernel)+galaxia_azul

print('sharpened laplaciano:',sharpened2)
plt.title('sharpened 2', fontsize=14)
plt.imshow(sharpened2)
plt.colorbar()
plt.show()


#imagen para informe tres

imagen_sharpened_graf=[sharpened1,sharpened2]

for i,f in enumerate(imagen_sharpened_graf):
    titulo=['shapened 1','sharpened 2']
    plt.subplot(1,2,i+1)
    plt.title(titulo[i])
    plt.imshow(f) 
plt.show()


# Pregunta 5:

kernel_1=np.array([[0,0,0],[-1,-1,0],[0,0,0]]) #primer filtro

kernel_2=np.zeros((5,5))
kernel_2[1:4,1:4]=np.ones((1,1))*(1/9)  #segundo filtro


kernel_3=np.array([[8,4,8],[4,2,4],[8,4,8]])*(1/20) #tercer filtro

filtro_1=ndimage.convolve(galaxia_azul,kernel_1)#-filter_galaxia_azul_gau
filtro_2=ndimage.convolve(galaxia_azul,kernel_2)#-filter_galaxia_azul_gau
filtro_3=ndimage.convolve(galaxia_azul,kernel_3)#-filter_galaxia_azul_gau

plt.title('filtro 1',fontsize=14)
plt.imshow(filtro_1)
plt.colorbar()
plt.show()

plt.title('filtro 2',fontsize=14)
plt.imshow(filtro_2)
plt.colorbar()
plt.show()

plt.title('filtro 3',fontsize=14)
plt.imshow(filtro_3)
plt.colorbar()
plt.show()

#imagen para informe cuatro

imagen_filtros_2=[filtro_1,filtro_2,filtro_3]

for i,j in enumerate(imagen_filtros_2):
    titulo=['filtro 1','filtro 2','filtro 3']
    plt.subplot(1,3,i+1)
    plt.title(titulo[i])
    plt.imshow(j)
plt.show()

#PARTE 2

# Pregunta 6:

# BIAS REAL

bias_real=copy.deepcopy(galaxia_azul)

bias_real[0:680,0:1200]=[70]

plt.title('bias real',fontsize=14)
plt.imshow(bias_real)
plt.colorbar()
plt.show()

#Ruido aleatorio 

bias_ruido=np.random.uniform(-10,10,size=np.shape(galaxia_azul))*1.5 

bias_ruido=bias_ruido.astype(int)

plt.title('bias ruido',fontsize=14)
plt.imshow(bias_ruido)
plt.colorbar()
plt.show()

#BIAS OBSERVADO 1 (gaussiana)


bias_gaussiano=np.random.normal(bias_ruido*2,size=np.shape(galaxia_azul)) 


bias_observado_1=bias_gaussiano+galaxia_azul

plt.title('bias observado 1',fontsize=14)
plt.imshow(bias_observado_1)
plt.colorbar()
plt.show()

#BIAS OBSERVADO 2

bias_observado_2=bias_ruido*3+galaxia_azul

plt.title('bias observado 2',fontsize=14)
plt.imshow(bias_observado_2)
plt.colorbar()
plt.show()

#BIAS OBSERVADO 3

bias_observado_3=bias_ruido*4+galaxia_azul
plt.title('bias observado 3',fontsize=14)
plt.imshow(bias_observado_3)
plt.colorbar()
plt.show()

#imagen para informe

bias_lista=[bias_real,bias_observado_1,bias_observado_2,bias_observado_3]

for i,j in enumerate(bias_lista):
    titulos=['bias real','bias observado 1','bias observado 2','bias observado 3']
    plt.subplot(2,2,i+1)
    plt.title(titulos[i],fontsize=9)
    plt.imshow(j)
plt.show()    

   

# pregunta 7:

# FLAT REAL

#circulo grande
new_image_azul=copy.deepcopy(galaxia_azul)
new_image_azul[0:675,0:1200]=[0]
new_image_azul[50:550,150:1000]=[50]
new_image_azul[100:500,200:900]=[100]
new_image_azul[200:400,400:800]=[180]

circulo_grande=ndimage.gaussian_filter(new_image_azul,sigma=100) 

#motas de polvo

circulo_grande[100:150,300:350]=[200]
circulo_grande[50:100,1000:1050]=[200]
circulo_grande[350:520,480:640]=[200]
circulo_grande[380:480,510:610]=[10]


flat_real=ndimage.gaussian_filter(circulo_grande,sigma=30)*0.5

plt.imshow(flat_real,cmap='gray')
plt.title('flat real',fontsize=14)
plt.colorbar()
plt.show()

#FLAT observado 

flat_observado_1=galaxia_azul*flat_real*0.8

plt.imshow(flat_observado_1)
plt.title('flat observado 1',fontsize=14)
plt.colorbar()
plt.show()

#FLAT observado 2

flat_observado_2=np.random.normal(flat_observado_1,size=np.shape(galaxia_azul))

plt.imshow(flat_observado_2)
plt.title('flat observado 2',fontsize=14)
plt.colorbar()
plt.show()

#Flat observado 3

flat_observado_3=galaxia_azul+(flat_real)*1.8

plt.imshow(flat_observado_3)
plt.title('Flat observado 3',fontsize=14)
plt.colorbar()
plt.show()

#imagen para informe 

flat_lista=[flat_real,flat_observado_1,flat_observado_2,flat_observado_3]

for i,j in enumerate(flat_lista):
    titulos=['flat real','flat observado 1','flat observado 2','flat observado 3']
    plt.subplot(2,2,i+1)
    plt.title(titulos[i],fontsize=9)
    plt.imshow(j)
plt.show()    


# Pregunta 8:

#Bias más flat

galaxia_bias_flat=(galaxia_azul*flat_real)+bias_real  

plt.imshow(galaxia_bias_flat)
plt.title('imagen bias+flat',fontsize=14)
plt.colorbar()
plt.show()

# Pregunta 9:

#funcion genera rayos

new_image_cosmic=copy.deepcopy(galaxia_bias_flat)

def rayos_cosmicos(rango,matriz): 
    init=np.random.randint(rango)
    finit=np.random.randint(rango)
    a=0
    b=0
    if init>finit:
        a=init
        b=finit
        init=b
        finit=a
    else:
        init=init
        finit=finit
        
    m=np.random.randint(rango)
    n=np.random.randint(rango)       

    for i in range (init,finit+1):
        matriz[i+n:i+2+n,i+m:i+2+m]=[80]
        matriz[i+1+n:i+3+n,i+m:i+2+m]=[80]
    
    return matriz
    

#imagen de rayos cosmicos 1 #FALTA HACERLO RANDOM

rayos_cosmicos(880,new_image_cosmic)
rayos_cosmicos(660,new_image_cosmic)
rayos_cosmicos(1100,new_image_cosmic)
rayos_cosmicos(900,new_image_cosmic)
rayos_cosmicos(70,new_image_cosmic)
rayos_cosmicos(900,new_image_cosmic)

plt.imshow(new_image_cosmic)
plt.title('rayos cosmicos 1',fontsize=14)
plt.colorbar()
plt.show()

#imagen de rayos cosmicos 2

new_image_cosmic_2=copy.deepcopy(galaxia_bias_flat)

rayos_cosmicos(1000,new_image_cosmic_2)
rayos_cosmicos(1200,new_image_cosmic_2)
rayos_cosmicos(800,new_image_cosmic_2)
rayos_cosmicos(900,new_image_cosmic_2)
rayos_cosmicos(30,new_image_cosmic_2)
rayos_cosmicos(900,new_image_cosmic_2)

plt.imshow(new_image_cosmic_2)
plt.title('rayos cosmicos 2',fontsize=14)
plt.colorbar()
plt.show()

#imagen de rayos cosmicos 3

new_image_cosmic_3=copy.deepcopy(galaxia_bias_flat)

rayos_cosmicos(100,new_image_cosmic_3)
rayos_cosmicos(10,new_image_cosmic_3)
rayos_cosmicos(500,new_image_cosmic_3)
rayos_cosmicos(400,new_image_cosmic_3)
rayos_cosmicos(100,new_image_cosmic_3)
rayos_cosmicos(900,new_image_cosmic_3)


plt.imshow(new_image_cosmic_3)
plt.title('rayos cosmicos 3',fontsize=14)
plt.colorbar()
plt.show()

#imagen para informe

lista_rayos=[new_image_cosmic,new_image_cosmic_2,new_image_cosmic_3]

for i,j in enumerate(lista_rayos):
    titulos=['rayos comiscos 1','rayos cosmicos 2','rayos cosmicos 3']
    plt.subplot(1,3,i+1)
    plt.title(titulos[i],fontsize=9)
    plt.imshow(j)
plt.show() 

# Pregunta 10: 

new_image_cosmic=np.where(new_image_cosmic<0,1,new_image_cosmic)

#ruido poisson 1

image_poisson_1=np.random.poisson(new_image_cosmic)

print('ruido poisson 1',image_poisson_1[0:20,0:20])

plt.imshow(image_poisson_1)
plt.title('ruido poisson 1',fontsize=14)
plt.colorbar()
plt.show()

#ruido poisson 2
new_image_cosmic_2=np.where(new_image_cosmic_2<0,1,new_image_cosmic_2)


image_poisson_2=np.random.poisson(new_image_cosmic_2)

plt.imshow(image_poisson_2)
plt.title('ruido poisson 2',fontsize=14)
plt.colorbar()
plt.show()

#ruido poisson 3

new_image_cosmic_3=np.where(new_image_cosmic_3<0,1,new_image_cosmic_3)

image_poisson_3=np.random.poisson(new_image_cosmic_3)


plt.imshow(image_poisson_3)
plt.title('ruido poisson 3',fontsize=14)
plt.colorbar()
plt.show()

#imagen para informe 

lista_poisson=[image_poisson_1,image_poisson_2,image_poisson_3]

for i,j in enumerate(lista_poisson):
    titulos=['ruido poisson 1','ruido poisson 2','ruido poisson 3']
    plt.subplot(1,3,i+1)
    plt.title(titulos[i],fontsize=9)
    plt.imshow(j)
plt.show() 

# Pregunta 12


# Reducción de imágenes. 
#programa mediana de matrices 

def Mediana(M1,M2,M3):
    M_new=np.zeros(np.shape(M1))
    for i in range (0,675):
        for j in range (0,1200):
            arreglo=np.array([M1[i][j],M2[i][j],M3[i][j]])
            M_new[i][j]=np.median(arreglo)
    return M_new
#quitar los ceros del flat

flat_real=np.where(flat_real<=0,1,flat_real)

print('flat sin ceros',flat_real[0:20,0:20])

#reduccion 

#Master


master_bias=Mediana(bias_ruido*0.5,bias_ruido*0.2,bias_ruido*0.1)*0.001
master_flat=Mediana(flat_real*0.5,flat_real*0.2,flat_real*0.1)

master_flat=np.where(master_flat<=1,1,master_flat)

print('flat sin ceros',master_flat[0:20,0:20])

#1

imagen_reduccion=(image_poisson_1-bias_ruido)/(flat_real)

print('imagen reduccion',imagen_reduccion[0:20,0:20])

plt.imshow(imagen_reduccion)
plt.title('imagen reducida',fontsize=14)
plt.colorbar()
plt.show()

#2

imagen_reduccion_2=(image_poisson_2-bias_ruido)/(flat_real)


plt.imshow(imagen_reduccion_2)
plt.title('imagen reducida 2',fontsize=14)
plt.colorbar()
plt.show()

#3

imagen_reduccion_3=(image_poisson_3-bias_ruido)/(master_flat)

plt.imshow(imagen_reduccion_3)
plt.title('imagen reducida 3',fontsize=14)
plt.colorbar()
plt.show()


# QUITAR RAYOS COSMICOS



#imagen final


imagen_lista_1=Mediana(imagen_reduccion,imagen_reduccion_2,imagen_reduccion_3)

plt.imshow(imagen_lista_1)
plt.title('imagen final',fontsize=14)
plt.colorbar()
plt.show()

#comparacion

imagen_comparacion=galaxia_azul-imagen_lista_1


plt.imshow(imagen_comparacion)
plt.title('imagen comparacion',fontsize=14)
plt.colorbar()
plt.show()









            








