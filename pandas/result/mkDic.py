# coding: utf-8
unique_cands= fec.cand_nm.unique()
tuc= tuple(unique_cands)
tuc
print(unique_cands.shape)
print(len(part_val))
unique_cands[2]
part_val= ['Republican']*6
part_val.append('Democrat')
part_val+= ['Republican']*6
part_val
# 안됨 parties= dict(keys= unique_cands, vars= part_val)
# 안됨 parties= dict(unique_cands=part_val)
parties= {}
for key, val in zip(tuc, part_val):
    parties[key]=val
parties
