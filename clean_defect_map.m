function winding_clean_out = clean_defect_map(winding_clean, min_separation)
% CLEAN_DEFECT_MAP  
%   1) Shrink each connected component to its centroid (single voxel)
%   2) Remove +/- pairs closer than min_separation pixels
%
% Parameters
% ----------
% winding_clean   : (H x W x Z) volume with 0, +0.5, -0.5
% min_separation  : scalar — remove pairs closer than this (pixels)

winding_clean_out = zeros(size(winding_clean));

% --- step 1: shrink each CC to its centroid ---
for charge = [0.5, -0.5]
    mask = (winding_clean == charge);
    CC = bwconncomp(mask, 26);   % 26-connectivity for 3D
    for k = 1:CC.NumObjects
        idx = CC.PixelIdxList{k};
        [r, c, z] = ind2sub(size(winding_clean), idx);
        % centroid rounded to nearest voxel
        cr = round(mean(r));
        cc = round(mean(c));
        cz = round(mean(z));
        winding_clean_out(cr, cc, cz) = charge;
    end
end

% --- step 2: remove +/- pairs closer than min_separation ---
[pr, pc, pz] = ind2sub(size(winding_clean_out), find(winding_clean_out ==  0.5));
[mr, mc, mz] = ind2sub(size(winding_clean_out), find(winding_clean_out == -0.5));

plus_coords  = [pr, pc, pz];
minus_coords = [mr, mc, mz];

remove_plus  = false(size(plus_coords,  1), 1);
remove_minus = false(size(minus_coords, 1), 1);

for i = 1:size(plus_coords, 1)
    for j = 1:size(minus_coords, 1)
        dist = norm(plus_coords(i,:) - minus_coords(j,:));
        if dist < min_separation
            remove_plus(i)  = true;
            remove_minus(j) = true;
        end
    end
end

% zero out the flagged defects
for i = find(remove_plus)'
    winding_clean_out(plus_coords(i,1), plus_coords(i,2), plus_coords(i,3)) = 0;
end
for j = find(remove_minus)'
    winding_clean_out(minus_coords(j,1), minus_coords(j,2), minus_coords(j,3)) = 0;
end

end