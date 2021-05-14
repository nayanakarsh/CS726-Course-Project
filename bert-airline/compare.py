# with open('sample.txt', 'r') as file1:
#     with open('./output/answer.txt', 'r') as file2:
#         same = set(file1).intersection(file2)

# same.discard('\n')

# count = 0
# with open('some_output_file.txt', 'w') as file_out:
#     for line in same:
#     	count = count + 1

# print(count)
from  sklearn.metrics import f1_score

i = 0
with open('expected_output.txt') as f:
        for i, l in enumerate(f):
            pass
length = i + 1
print("length of file = ",length)
        

file1 = open('expected_output.txt')
  
# read the content of the file opened
content1 = file1.read()
content1=content1.split('\n')
content1=content1[:-1]


file2 = open('./output/answer.txt')
content2 = file2.read()
content2=content2.split('\n')

ans=f1_score(content1,content2, average='macro')
print("F1score= "+str(ans))
# count = 0
# for i in range(length):
# 	if(content1[i]==content2[i]):
# 		count = count + 1

# print("number of matches = ",count)
# print("Accuracy = ",count/length)
