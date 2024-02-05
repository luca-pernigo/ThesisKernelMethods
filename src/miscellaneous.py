
# find index i s.t. vec[i]<x<vec[i+1]
def find(x, vec):
    l=0
    r=vec.size-1
    while True:
        m=int((l+r)/2)
        
        if x>vec[m]:
            l=m
        elif x<vec[m]:
            r=m
        
        if r-l==1:
            break

    return l

# def find(x, vec):
#     for i in range(len(vec)):
#         if x>vec[i] and x<vec[i+1]:
#             ans=i
#             break
#     return i
