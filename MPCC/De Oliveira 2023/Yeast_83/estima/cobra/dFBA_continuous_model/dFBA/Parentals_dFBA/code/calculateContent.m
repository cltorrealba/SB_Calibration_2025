%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Ptot,Ctot] = calculateContent(model)
%
% Benjam�n J. S�nchez
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ptot,Ctot,Rtot] = calculateContent(model)

% MW aminoacids [g/mol]:
aas = {'s_0404[c]'	89.09       % A     Alanine         ala
    's_0542'	121.16      % C     Cysteine        cys
    's_0432'	133.11      % D     Aspartic acid   asp
    's_0748'	147.13      % E     Glutamic acid   glu
    's_1314'	165.19      % F     Phenylalanine   phe
    's_0757'	75.07       % G     Glycine         gly
    's_0832'	155.15      % H     Histidine       his
    's_0847'	131.17      % I     Isoleucine      ile
    's_1099'	146.19      % K     Lysine          lys
    's_1077'	131.17      % L     Leucine         leu
    's_1148'	149.21      % M     Methionine      met
    's_0430'	132.12      % N     Asparagine      asn
    's_1379'	115.13      % P     Proline         pro
    's_0747'	146.14      % Q     Glutamine       gln
    's_0428'	174.2       % R     Arginine        arg
    's_1428'	105.09      % S     Serine          ser
    's_1491'	119.12      % T     Threonine       thr
    's_1561'	117.15      % V     Valine          val
    's_1527'	204.23      % W     Tryptophan      trp
    's_1533'	181.19};    % Y     Tyrosine        tyr

% MW carbohidrates [g/mol]:
carbs = {'s_0001'	180.16      % (1->3)-beta-D-glucan
    's_0004'	180.16      % (1->6)-beta-D-glucan
    's_0509'	221.21      % chitin
    's_0773'	180.16      % glycogen
    's_1107'	180.16      % mannan
    's_1520'	342.296};	% trehalose


% CMP [cytoplasm]	s_0526[c] 323.1965
% GMP [cytoplasm]	s_0782[c] 363.22
% UMP [cytoplasm]	s_1545[c] 324.1813

rnas={'s_0423' 347.2212  %AMP
    's_0526' 323.1965  %CMP
    's_0782' 363.22    %GMP
    's_1545' 324.1813};  %UMP}

%Initialize protein and carb content:
Ptot = 0;
Ctot = 0;
Rtot=0;

%Count protein/carb content in the corresponding pseudo-rxn:
protPos = strcmp(model.metNames,'protein');
carbPos = strcmp(model.metNames,'carbohydrate');
rnaPos = strcmp(model.metNames,'RNA');
protRxn = model.S(protPos,:) == 1;
carbRxn = model.S(carbPos,:) == 1;
rnaRxn = model.S(rnaPos,:) == 1;

for i = 1:length(model.mets)
    
    posP = strcmp(aas(:,1),model.mets{i});
    posC = strcmp(carbs(:,1),model.mets{i});
    posR = strcmp(rnas(:,1),model.mets{i});
    
    if sum(posP) == 1
        Sprot = abs(model.S(i,protRxn));            % mmol/gDW
        Ptot  = Ptot + Sprot*aas{posP,2}/1000;      % mmol/gDW * g/mmol = g/gDW
%      fprintf('%s\t%f\t%f\t%f\n',aas{posP,1},aas{posP,2},full(Sprot),full(Sprot*aas{posP,2}));
    elseif sum(posC) == 1
        Scarb = abs(model.S(i,carbRxn));            % mmol/gDW
        Ctot  = Ctot + Scarb*carbs{posC,2}/1000;    % mmol/gDW * g/mmol = g/gDW
    elseif sum(posR) == 1
        Srna = abs(model.S(i,rnaRxn));            % mmol/gDW
        Rtot  = Rtot + Srna*rnas{posR,2}/1000;    % mmol/gDW * g/mmol = g/gDW
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%