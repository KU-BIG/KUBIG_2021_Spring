
# coding: utf-8

# ## Pythos as a calculator
# 기본적인 연산 커맨드
# [ + : 덧셈, - : 뺄셈, / : 나누기, % : 나머지, ** : n제곱 ]

# In[1]:

print(5 + 3)
print(5 - 3)
print(5 / 3)
print(5 % 3)
print(5 ** 3)
# 초기금액 100원, 연이율 10%, 7년 후
print(100*1.1**7)


# ## Variables and Types
# Variable 설정과 파이썬이 가지는 Types에 대하여
# * float : 정수 부분과 소수 부분 모두 존재
# * int : 정수
# * str : 텍스트
# * bool : True or False, 데이터 필터링 작업에 주로 활용
# * 데이터 타입에 따라 코드의 작동 방식이 달라진다.
# * Ex) 숫자에서 + 는 더하기 연산, 텍스트에서 + 는 두 텍스트를 이어 붙이기.

# In[2]:

# 초기금액 100원, 연이율 10%, 7년 후_변수 설정
savings = 100
growth_multiplier = 1.1
result = savings * growth_multiplier**7
print(result)


# In[3]:

# type() 함수를 이용한 Types Finding
a = 3.141592
b = 3
c = 'python'
d = True
print(type(a))
print(type(b))
print(type(c))
print(type(d))


# In[4]:

# 데이터 타입과 코드의 작동 방식
# Ex) 숫자에서 + 는 더하기 연산, 텍스트에서 + 는 두 텍스트를 이어 붙이기.
a = 1
b = 2
print(a+b)
c = 'c'
d = 'd'
print(c+d)


# In[5]:

# 데이터 타입의 통일과 데이터 타입 변환
print("I started with $" + str(savings) + " and now have $" + str(result) + ".")
pi_string = "3.141592"
print(type(pi_string))
pi_float = float(pi_string)
print(type(pi_float))

