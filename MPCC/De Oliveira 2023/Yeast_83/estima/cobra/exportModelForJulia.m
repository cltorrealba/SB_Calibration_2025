function exportModelForJulia(model, outDir)
    % exportModelForJulia(model, outDir)
    % Genera:
    %   - S.csv, lb.csv, ub.csv
    %   - rxn_ids.txt (una reacción por línea)
    %   - met_ids.txt (un metabolito por línea)

    if nargin < 2
        outDir = pwd;
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % 1) Matriz S en formato denso
    S_full = full(model.S);
    writematrix(S_full, fullfile(outDir, 'S.csv'));

    % 2) Bounds
    writematrix(model.lb(:), fullfile(outDir, 'lb.csv'));
    writematrix(model.ub(:), fullfile(outDir, 'ub.csv'));

    % 3) IDs de reacciones y metabolitos (texto plano)
    fid = fopen(fullfile(outDir, 'rxn_ids.txt'), 'w');
    for i = 1:numel(model.rxns)
        fprintf(fid, '%s\n', model.rxns{i});
    end
    fclose(fid);

    fid = fopen(fullfile(outDir, 'met_ids.txt'), 'w');
    for i = 1:numel(model.mets)
        fprintf(fid, '%s\n', model.mets{i});
    end
    fclose(fid);
end
