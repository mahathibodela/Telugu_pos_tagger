import functools
a = ['2','3','0']

def compare(s1,s2):
    if (s1+s2) > (s2+s1):
        return 1
    return -1

a.sort(key = functools.cmp_to_key(compare))
print(a)