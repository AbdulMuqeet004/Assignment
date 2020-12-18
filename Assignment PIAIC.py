#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
arr=np.array([1,2,3])  
arr 


# In[3]:


x=np.array([[1,2],[3,4]]) 
y=np.array([[12,30]])  
z=np.concatenate((x,y))  
z


# In[4]:


a=np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])  
b=np.array([[11, 21, 31], [42, 52, 62], [73, 83, 93]])  
c=np.append(a,b)  
c 


# In[5]:


x=np.arange(12)  
y=np.reshape(x, (4,3))  
x  
y


# In[6]:


a=np.array([0.4,0.5])  
b=np.sum(a)  
b 


# In[7]:


a=np.random.rand(5,2)  
a 


# In[8]:


a=np.zeros(6)  
a


# In[9]:


a=np.array([2, 4, 6, 3**8])  
a  
b=np.log(a)  
b  
c=np.log2(a)  
c  
d=np.log10(a)  
d 


# In[10]:


a=np.arange(12)  
b=np.where(a<6,a,5*a)  
b 


# In[11]:


a=np.array([456,11,63])  
a  
b=np.argsort(a)  
b


# In[12]:


a= np.arange(6).reshape((2,3))  
a  
b=np.transpose(a)  
b


# In[13]:


a = np.array([[1, 2], [3, 4]])  
b=np.mean(a)  
b  
x = np.array([[5, 6], [7, 34]])  
y=np.mean(x)  
y 


# In[14]:


a=np.unique([1,2,3,4,3,6,2,4])  
a  


# In[16]:


a = np.array([[11, 21], [31, 41]])  
b=a.tolist()  
a  
b 


# In[17]:


a = [[1, 2], [4, 1]]  
b = [[4, 11], [2, 3]]  
c=np.dot(a, b)  
c


# In[18]:


from io import StringIO  
c = StringIO(u"0 1\n2 3")  
c  
np.loadtxt(c)  


# In[19]:


a = np.arange(12)  
np.clip(a, 3, 9, out=a)  
a 


# In[20]:


a = np.array([[1,4,7], [2,5,8],[3,6,9]])  
b=a.flatten()  
b 


# In[21]:


na, nb = (5, 3)  
a = np.linspace(1, 2, na)  
b = np.linspace(1, 2, nb)  
xa, xb = np.meshgrid(a, b, sparse=True)  
xa  
xb 


# In[22]:


a=np.array([[1,4,7,10],[2,5,8,11]])  
b=np.std(a)  
b


# In[23]:


x = np.array([11, 21, 41, 71, 1, 12, 33, 2])  
y = np.diff(x)  
x  
y 


# In[24]:


x = np.empty([3, 3], dtype=float)  
x 


# In[25]:


a=np.histogram([1, 5, 2], bins=[0, 1, 2, 3])  
a 


# In[26]:


x=np.array([[1,4,2,3],[9,13,61,1],[43,24,88,22]])  
x  
y=np.sort(x)  
y


# In[27]:


data = list(range(1,6))  
output=np.average(data)  
data  
output 


# In[28]:


x = [1, 3, 2, 5, 4]  
y = np.pad(x, (3, 2), 'edge')  
y


# In[29]:


x = np.array([[1, 3, 5], [11, 35, 56]])  
y=np.ravel(x)  
y 


# In[30]:


arr = [0, 0.3, -1]   
print ("Input array : \n", arr)   
  
arccos_val = np.arccos(arr)   
print ("\nInverse cos values : \n", arccos_val) 


# In[31]:


arr = np.logspace(10, 20, num = 5,base = 2, endpoint = True)  
print("The array over the given range is ",arr)


# In[32]:


import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
degval = np.degrees(arr)   
print ("\n Degree value : \n", degval)


# In[33]:


arr = np.arange(0,10,2,float)  
print(arr)


# In[34]:


l=[1,2,3,4,5,6,7]  
  
a = np.asarray(l);  
  
print(type(a))  
  
print(a)


# In[35]:


arr = np.linspace(10, 20, 5)  
print("The array over the given range is ",arr)


# In[36]:


arr = [0.23, 0.09, 1.2, 1.24, 9.99]  
  
print("Input array:",arr)  
  
r_arr = np.fix(arr)  
  
print("Output array:",r_arr)


# In[37]:


arr = [0, 30, 60, 90 ]   
print ("Input array : \n", arr)   
  
radval = np.radians(arr)   
print ("\n Radian value : \n", radval)


# In[38]:


x = np.arange(20).reshape(4,5) + 7  
x  
y=np.argmax(a)  
y 


# In[39]:


base = [10,2,5,50]  
per= [3,10,23,6]  
  
print("Input base array:",base)  
print("Input perpendicular array:",per)  
  
hyp = np.hypot(base,per)  
  
print("hypotenuse ",hyp)


# In[40]:


a = 10  
b = 12  
  
print("binary representation of a:",bin(a))  
print("binary representation of b:",bin(b))  
print("Bitwise-and of a and b: ",np.bitwise_and(a,b))


# In[41]:


print("Concatenating two string arrays:")  
print(np.char.add(['welcome','Hi'], [' to Javatpoint', ' read python'] )) 


# In[42]:


a = np.array([[10,2,3],[4,5,6],[7,8,9]])  
  
print("Sorting along the columns:")  
print(np.sort(a))  
  
print("Sorting along the rows:")  
print(np.sort(a, 0))  
  
data_type = np.dtype([('name', 'S10'),('marks',int)])  
  
arr = np.array([('Mukesh',200),('John',251)],dtype = data_type)  
  
print("Sorting data ordered by name")  
  
print(np.sort(arr,order = 'name'))


# In[43]:


a = np.array([[1,2,3,4],[2,4,5,6],[10,20,39,3]])  
print("Printing array:")  
print(a);  
print("Iterating over the array:")  
for x in np.nditer(a):  
    print(x,end=' ') 


# In[44]:


a = np.array([[1,2,3,4],[9,0,2,3],[1,2,3,19]])  
  
print("Original Array:\n",a)  
  
print("\nID of array a:",id(a))  
  
b = a   
  
print("\nmaking copy of the array a")  
  
print("\nID of b:",id(b))  
  
b.shape = 4,3;  
  
print("\nChanges on b also reflect to a:")  
print(a) 


# In[45]:


array1=np.array([[1,2,3],[4,5,6],[7,8,9]],ndmin=3)  
array2=np.array([[9,8,7],[6,5,4],[3,2,1]],ndmin=3)  
result=np.multiply(array1,array2)  
result


# In[46]:


a = np.array([[10,2,3],[4,5,6],[7,8,9]])  
  
print("Sorting along the columns:")  
print(np.sort(a))  
  
print("Sorting along the rows:")  
print(np.sort(a, 0))  
  
data_type = np.dtype([('name', 'S10'),('marks',int)])  
  
arr = np.array([('Mukesh',200),('John',251)],dtype = data_type)  
  
print("Sorting data ordered by name")  
  
print(np.sort(arr,order = 'name')) 


# In[47]:


d = np.int32(i4)  
print(d) 


# In[48]:


import numpy.matlib    
    
print(numpy.matlib.zeros((3,3))) 


# In[49]:


import numpy.matlib    
    
print(numpy.matlib.eye(n=3,m=3,k=0,dtype=int))


# In[ ]:




