% PSR = (Fmax - mean(response)) / std(response)
function psr = cal_psr(response)

response_temp = response;

Fmax = max(response_temp(:));

psr = (Fmax - mean2(response_temp)) / std2(response_temp);

end