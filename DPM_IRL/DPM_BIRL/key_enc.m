function key1_char =  key_enc(key1)

[m,n] = size(key1);
key1_char=repmat({char(ones(1,16*n))},m,1);
tmp = num2hex(key1);
for ii=1:m
key1_char{ii,:} = reshape((tmp(ii:m:end,:).'),1,16*n);
key1_char = char(key1_char);
end

end