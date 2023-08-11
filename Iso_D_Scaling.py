import math

# Real newDgb =
#         Dgb * (1.5 * _GBwidth / _int_width) + _Dbulk * (1 - 1.5 * (_GBwidth / _int_width));
#     Real newDsurf = Dsurf * (1.5 * _surf_thickness / _int_width) +
#                     _Dbulk * (1 - 1.5 * (_surf_thickness / _int_width));
dgb_coeff = float(input('D_gb: '))
ds_coeff = float(input('D_s: '))
iw = 20
Db = 8.33e9 * math.exp(-3.608 / 8.617343e-5 / 1600)
dgb = Db * dgb_coeff
ds = Db * ds_coeff
print('Bulk D = ',Db)
print(' ')

scaled_gb = dgb * (1.5*0.5/iw) + Db * (1 - 1.5*(0.5/iw))
scaled_s = ds * (1.5*0.5/iw) + Db * (1 - 1.5*(0.5/iw))

pre_gb = (dgb - Db * (1 - 1.5*(0.5/iw)) ) / (1.5*0.5/iw)
pre_s = (ds - Db * (1 - 1.5*(0.5/iw)) ) / (1.5*0.5/iw)

input_dgb = pre_gb / Db
input_ds = pre_s / Db

def Dgb_of_Ds(Ds_coeff,Dgb_div_Ds,delta):
    ds = Db * Ds_coeff
    scaled_s = ds * (1.5*0.5/iw) + Db * (1 - 1.5*(0.5/iw))
    # Delta here is assuming it only applies to GB and not Surf
    scaled_gb = scaled_s * Dgb_div_Ds / delta
    back_gb = (scaled_gb - Db * (1 - 1.5*(0.5/iw)) ) / (1.5*0.5/iw)
    gb_input = back_gb / Db
    return gb_input



print('With input values, the scaled Dgb and Ds we use is:')
print(scaled_gb)
print(scaled_s)
print('or just x times bulk:')
print(scaled_gb/Db)
print(scaled_s/Db)
print(' ')
print('Input File values for delta*Dgb = Ds')
print('Dgb: ',Dgb_of_Ds(ds_coeff,1,iw))
print("Ds: ",ds_coeff)
print(' ')
print('With input as the scaled value, the input multiplier needs to be:')
print(input_dgb)
print(input_ds)
