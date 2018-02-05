function [V,D, meanSample] = pca2(sampleMat,nPercent)

nSamples = size(sampleMat,1);
nDim = size(sampleMat,2);

% Ortalama vektoru bul
meanSample = mean(sampleMat,1);

% sampleMat'taki tum elemanlardan ortalamayi cikararak normalizasyon yap
sampleMat = sampleMat - repmat(meanSample, nSamples, 1);

% Eger boyut sayisi ornek sayisindan coksa, hizli olmasi icin
% sampleMat'in transpozesini al
if nDim > nSamples
    sampleMat = sampleMat.';
end

% Dagilimin kovaryansini bul
C = sampleMat.' * sampleMat ./ nSamples;

% Ozdegerler (D) ve ozvektorleri (V) bul
% Hatirla ki: A*V = V*D ve V'nin her sutunu bir ozvektor
[V,D] = eig(C);
D = diag(D);

% Ozdegerleri buyukten kucuge siralamak gerekli. Matlab tersinden
% buldugu icin degerleri ters dondur.
D = flipud(D); % Sutun vektorunu altust et
V = fliplr(V); % Matrisin sagdan sola aynasini al

% Eger hizlandirma kullanildiysa Buldugumuz sutunlardan ozvektor bulup
% ozvektorleri normalize etmeliyiz. Ozdegerler aynidir. Bak: Eckart-Young Teoremi
if nDim > nSamples
    V = sampleMat * V;
    for i = 1:nSamples
        normV = norm(V(:,i));
        V(:,i) = V(:,i) ./ normV;
    end
end



for i = 1 : size(D,1)
 energy = sum(D(1:i))/sum (D);
 if energy > nPercent
        energyInd = i;
        D = D(1:energyInd);
        V = V(:,1:energyInd);
        break
 end
end
