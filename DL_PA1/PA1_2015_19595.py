from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import math

class c_read_file:
    def read_input_data(file_name):
        f = open(file_name,'r')
        all_lines = f.readlines()
        global  number_of_triangle
        number_of_triangle = len(all_lines)
        f.close
        return all_lines

class c_triangle_2D:
    def init_triangle_info(all_lines,number_of_lines):
        global triangle
        triangle = []
        i = 0
      
        for each_line in all_lines:
            line_splite = each_line.split(' ');
            triangle.append([])

            for k in range(0,len(line_splite)-1):
                 triangle[i].append(int(line_splite[k+1]))
                 k +=1
            i+=1
            
class c_calculator:
    def cal_area_of_triangle(a):
        area =abs( 0.5*(a[0]*a[3]+a[2]*a[5]+a[4]*a[1])-0.5*(a[2]*a[1]+a[4]*a[3]+a[0]*a[5]))  
        return area

    def cal_average(list):
        average_of_area = sum(list) / number_of_triangle
        return average_of_area

    def cal_standard_deviation(list):
        average = sum(list)/ number_of_triangle
        deviations = []
        sum_of_deviations = 0
        for i in range(0,number_of_triangle):
            deviations.append(list[i]-average)
            sum_of_deviations += deviations[i]*deviations[i]
        standard_deviation = np.sqrt(sum_of_deviations / (number_of_triangle-1))
        return standard_deviation

class c_draw_histogram:
    def draw_histogram(area_of_triangle):
        bins = np.arange(0,12,2)
        hist, bins = np.histogram(area_of_triangle, bins)
        print(hist)
        print(bins)
        plt.hist(area_of_triangle, bins,rwidth = 0.8)
        plt.xlabel('AREA', fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
       
        return

class c_write_file:
    def write_output_data(area_of_triangle,mean_of_area,standard_deviation):
        f = open("output_data.txt",'w')
        f.write("output\n")
        for i in range(0,number_of_triangle):
            f.write("Triangle[%d] = %f"%(i,area_of_triangle[i]))
            f.write('\n')

        f.write("Average of Area : "+ str(mean_of_area)+'\n')
        f.write("Standard Deviation of Area : "+ str(standard_deviation)+'\n')
        f.write('\n')
        f.write("Frequency Distribution of Area\n")
        bins = np.arange(0,12,2)
        hist, bins = np.histogram(area_of_triangle, bins)
        f.write(str(bins[0])+"~"+str(bins[1])+" : "+ str(hist[0])+'\n')
        f.write(str(bins[1])+"~"+str(bins[2])+" : "+ str(hist[1])+'\n')
        f.write(str(bins[2])+"~"+str(bins[3])+" : "+ str(hist[2])+'\n')
        f.write(str(bins[3])+"~"+str(bins[4])+" : "+ str(hist[3])+'\n')
        f.write(str(bins[4])+"~"+str(bins[5])+" : "+ str(hist[4])+'\n')
        f.close

    def write_vector_of_affine_transformed_triangle(transformed_triangle):
        f = open("output_data_2.txt",'w')
        f.write("output : affine transformed triangle data \n")
        for i in range(0,number_of_triangle):
           f.write("[%d] /  "%i)
           for k in range(0,6,2):
                 f.write("["+str(transformed_triangle[i][k])+" , "+str(transformed_triangle[i][k+1])+"]   ")
           f.write('\n')
        f.close


class c_affine_transform:
    def calculate_affine_transform(x,y, matrix , vector):
        v1 = round(matrix[0]*x + matrix[1]*y + vector[0],2)
        v2 = round( matrix[2]*x + matrix[3]*y + vector[1],2)
        return [v1,v2]

class c_matrix:
    def set_matrix(theta):
        a11 = math.cos(theta)
        a12 = -1*math.sin(theta)
        a21 = math.sin(theta)
        a22 = math.cos(theta)
        return [a11,a12,a21,a22]


class c_vector:
    def set_vector(a,b):
        return [a,b]

class c_draw_triangle:
    def draw_triangles(triangle, location):
        ax = fig.add_subplot(gs[location[0], location[1]])
        for i in range(0,number_of_triangle):
            x = [triangle[i][0],triangle[i][2],triangle[i][4],triangle[i][0]]
            y = [triangle[i][1],triangle[i][3],triangle[i][5],triangle[i][1]]
            ax.plot(x,y)
        


######### Problem 1 #####################

file_name = "input_data.txt"
number_of_triangle = 0
triangle = []
area_of_triangle = []
all_lines = c_read_file.read_input_data(file_name);
c_triangle_2D.init_triangle_info(all_lines,number_of_triangle);

for i in range(0,number_of_triangle):
    area_of_triangle.append(c_calculator.cal_area_of_triangle(triangle[i]))
    print(area_of_triangle[i])

mean_of_area= c_calculator.cal_average(area_of_triangle)

standard_deviation = c_calculator.cal_standard_deviation(area_of_triangle)
c_draw_histogram.draw_histogram(area_of_triangle)
c_write_file.write_output_data(area_of_triangle,mean_of_area,standard_deviation)


######### Problem 2 #####################
affine_transformed_triangle = []
theta = 0.25*math.pi
matrix = c_matrix.set_matrix(theta)
vector = c_vector.set_vector(3,3)

for i in range(0,number_of_triangle):
    affine_transformed_triangle.append([])

    transform =c_affine_transform.calculate_affine_transform(triangle[i][0],triangle[i][1],matrix,vector)
    affine_transformed_triangle[i].append(transform[0])
    affine_transformed_triangle[i].append(transform[1])

    transform2 =c_affine_transform.calculate_affine_transform(triangle[i][2],triangle[i][3],matrix,vector)
    affine_transformed_triangle[i].append(transform2[0])
    affine_transformed_triangle[i].append(transform2[1])

    transform3 =c_affine_transform.calculate_affine_transform(triangle[i][4],triangle[i][5],matrix,vector)
    affine_transformed_triangle[i].append(transform3[0])
    affine_transformed_triangle[i].append(transform3[1])
    
c_write_file.write_vector_of_affine_transformed_triangle(affine_transformed_triangle)

fig = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=1, ncols=2)
location1 = [0,0]
location2 = [0,1]
c_draw_triangle.draw_triangles(triangle,location1)
c_draw_triangle.draw_triangles(affine_transformed_triangle,location2)
plt.show()
        