for i=1:length(model.mets)
   model.metNames(i)=strcat(model.metNames(i),'[',model.comps(model.metComps(i)),']'); 
end