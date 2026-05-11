function skel_clean = disclinationClean(skel, options)
% DISCLINATIONCLEAN  Clean a 3D disclination skeleton by per-component
%                    dilation → Gaussian smoothing → threshold → re-skeletonisation.
%
% Usage
% -----
%   % Minimal:
%   skel_clean = disclinationClean(skel)
%
%   % With options:
%   skel_clean = disclinationClean(skel, ...
%       DilateRadius  = 4,  ...
%       GaussSigma    = 3,  ...
%       GaussThresh   = 0.3, ...
%       MinBranchLen  = 10, ...
%       Pad           = 10, ...
%       MinVoxels     = 20, ...
%       Verbose       = true)
%
% Required argument
% -----------------
%   skel        logical (H, W, D)   Input skeleton (e.g. from bwskel).
%
% Name-value options                  Default   Description
% ──────────────────────────────────────────────────────────────────────────────
%   DilateRadius  (1,1) double         4        Dilation radius [voxels] before
%                                               Gaussian smoothing. Gives the
%                                               filter something to work with on
%                                               the 1-voxel-thin skeleton.
%   GaussSigma    (1,1) double         3        Gaussian sigma [voxels].
%                                               Larger = smoother result but
%                                               small loops may shrink/disappear.
%   GaussThresh   (1,1) double         0.3      Threshold on normalised smoothed
%                                               volume [0-1]. Lower keeps more,
%                                               higher is more aggressive.
%   MinBranchLen  (1,1) double         10       MinBranchLength passed to bwskel
%                                               after re-skeletonisation to prune
%                                               remaining short fingers.
%   Pad           (1,1) double         10       Bounding-box padding [voxels].
%                                               Prevents edge effects during
%                                               smoothing of each component.
%   MinVoxels     (1,1) double         20       Skip components with fewer
%                                               skeleton voxels than this.
%   Verbose       (1,1) logical        true     Print per-component progress.

    arguments
        skel                    (:,:,:) logical
        options.DilateRadius    (1,1)   double  = 4
        options.GaussSigma      (1,1)   double  = 3
        options.GaussThresh     (1,1)   double  = 0.3
        options.MinBranchLen    (1,1)   double  = 10
        options.Pad             (1,1)   double  = 10
        options.MinVoxels       (1,1)   double  = 20
        options.Verbose         (1,1)   logical = true
    end

    % ── validate ─────────────────────────────────────────────────────────────
    assert(options.GaussThresh > 0 && options.GaussThresh < 1, ...
           'GaussThresh must be in (0, 1), got %.3f', options.GaussThresh)
    assert(options.DilateRadius >= 1, ...
           'DilateRadius must be >= 1, got %.1f', options.DilateRadius)

    [H, W, D] = size(skel);
    skel_clean = false(H, W, D);

    % ── connected components ──────────────────────────────────────────────────
    CC   = bwconncomp(skel, 26);
    se   = strel('sphere', round(options.DilateRadius));
    pad  = round(options.Pad);

    n_kept    = 0;
    n_skipped = 0;
    t0        = tic;

    if options.Verbose
        fprintf('disclinationClean: %d components found\n', CC.NumObjects)
    end

    for k = 1 : CC.NumObjects

        % ── skip small components ─────────────────────────────────────────────
        if numel(CC.PixelIdxList{k}) < options.MinVoxels
            n_skipped = n_skipped + 1;
            continue
        end

        % ── isolate component ─────────────────────────────────────────────────
        comp = false(H, W, D);
        comp(CC.PixelIdxList{k}) = true;

        % ── tight padded bounding box ─────────────────────────────────────────
        props = regionprops3(comp, 'BoundingBox');
        bb    = props.BoundingBox;   % [x0 y0 z0 xw yw zw]

        x1 = max(1, floor(bb(1))        - pad);
        y1 = max(1, floor(bb(2))        - pad);
        z1 = max(1, floor(bb(3))        - pad);
        x2 = min(W, ceil(bb(1)+bb(4))  + pad);
        y2 = min(H, ceil(bb(2)+bb(5))  + pad);
        z2 = min(D, ceil(bb(3)+bb(6))  + pad);

        crop = comp(y1:y2, x1:x2, z1:z2);

        % ── dilate → smooth → normalise → threshold ───────────────────────────
        thick  = imdilate(crop, se);
        smooth = imgaussfilt3(single(thick), options.GaussSigma);
        vmax   = max(smooth(:));
        if vmax < eps
            n_skipped = n_skipped + 1;
            continue
        end
        smooth = smooth / vmax;
        mask   = smooth > options.GaussThresh;

        if ~any(mask(:))
            n_skipped = n_skipped + 1;
            continue
        end

        % ── re-skeletonise with finger pruning ────────────────────────────────
        s = bwskel(mask, 'MinBranchLength', round(options.MinBranchLen));

        if ~any(s(:))
            n_skipped = n_skipped + 1;
            continue
        end

        % ── write back into full volume ───────────────────────────────────────
        skel_clean(y1:y2, x1:x2, z1:z2) = skel_clean(y1:y2, x1:x2, z1:z2) | s;
        n_kept = n_kept + 1;

        if options.Verbose && mod(k, max(1, floor(CC.NumObjects/10))) == 0
            fprintf('  %d / %d components  (%.0fs elapsed)\n', ...
                    k, CC.NumObjects, toc(t0))
        end
    end

    if options.Verbose
        fprintf('Done in %.1fs  |  kept: %d  |  skipped: %d  |  voxels: %d\n', ...
                toc(t0), n_kept, n_skipped, sum(skel_clean(:)))
    end
end