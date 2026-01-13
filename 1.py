
for i in range(256):
    for j in range(256):
        b = i.to_bytes() + j.to_bytes()
        try:
            b.decode('utf-16')
        except UnicodeDecodeError:
            print(b)
            exit(0)





# b = int(2).to_bytes()
# c=  int(2).to_bytes()
# print(b+c)