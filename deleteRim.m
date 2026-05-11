function isoSurfaces =  deleteRim(isoSurfaces, margin)
    isoSurfaces(1:margin, :, :) = false;
    isoSurfaces(end-margin+1:end, :, :) = false;
    isoSurfaces(:, 1:margin, :) = false;
    isoSurfaces(:, end-margin+1:end, :) = false;
    isoSurfaces(:, :, 1:margin) = false;
    isoSurfaces(:, :, end-margin+1:end) = false;
end