function skel_clean = disclinationClean_chordRemoval(skel, options)
% DISCLINATIONCLEAN  Clean a 3D disclination skeleton by per-component
%                    dilation → Gaussian smoothing → threshold → re-skeletonisation,
%                    followed by optional removal of chord fingers that cross loops.
%
% Usage
% -----
%   skel_clean = disclinationClean(skel)
%
%   skel_clean = disclinationClean(skel, ...
%       DilateRadius   = 4,    ...
%       GaussSigma     = 3,    ...
%       GaussThresh    = 0.3,  ...
%       MinBranchLen   = 10,   ...
%       Pad            = 10,   ...
%       MinVoxels      = 20,   ...
%       RemoveChords   = true, ...
%       MaxChordLength = 30,   ...
%       Verbose        = true)
%
% Required argument
% -----------------
%   skel          logical (H,W,D)   Input skeleton (e.g. from bwskel).
%
% Name-value options                   Default   Description
% ─────────────────────────────────────────────────────────────────────────────
%   DilateRadius   (1,1) double          4       Dilation radius [vox] before smoothing.
%   GaussSigma     (1,1) double          3       Gaussian sigma [vox].
%   GaussThresh    (1,1) double          0.3     Threshold on normalised smoothed volume.
%   MinBranchLen   (1,1) double          10      Finger pruning length passed to bwskel.
%   Pad            (1,1) double          10      Bounding-box padding [vox].
%   MinVoxels      (1,1) double          20      Skip components smaller than this.
%   RemoveChords   (1,1) logical         true    Remove fingers that cross a loop
%                                                at 2 contact points.
%   MaxChordLength (1,1) double          30      Only remove chords shorter than this
%                                                [vox]. Protects long loop arcs from
%                                                being mistaken for chords.
%   Verbose        (1,1) logical         true    Print progress.

    arguments
        skel                        (:,:,:) logical
        options.DilateRadius        (1,1)   double  = 4
        options.GaussSigma          (1,1)   double  = 3
        options.GaussThresh         (1,1)   double  = 0.3
        options.MinBranchLen        (1,1)   double  = 10
        options.Pad                 (1,1)   double  = 10
        options.MinVoxels           (1,1)   double  = 20
        options.RemoveChords        (1,1)   logical = true
        options.MaxChordLength      (1,1)   double  = 100
        options.Verbose             (1,1)   logical = true
    end

    assert(options.GaussThresh > 0 && options.GaussThresh < 1, ...
           'GaussThresh must be in (0,1), got %.3f', options.GaussThresh)

    [H, W, D]  = size(skel);
    skel_clean = false(H, W, D);
    se         = strel('sphere', round(options.DilateRadius));
    pad        = round(options.Pad);
    kernel26   = ones(3,3,3,'single');  kernel26(2,2,2) = 0;

    % ── Step 1: per-component smooth + re-skeletonise ─────────────────────────
    CC       = bwconncomp(skel, 26);
    n_kept   = 0;  n_skipped = 0;
    t0       = tic;

    if options.Verbose
        fprintf('disclinationClean: %d components\n', CC.NumObjects)
    end

    for k = 1 : CC.NumObjects
        if numel(CC.PixelIdxList{k}) < options.MinVoxels
            n_skipped = n_skipped + 1;
            continue
        end

        comp = false(H, W, D);
        comp(CC.PixelIdxList{k}) = true;

        props = regionprops3(comp, 'BoundingBox');
        bb    = props.BoundingBox;
        x1 = max(1, floor(bb(1))       - pad);
        y1 = max(1, floor(bb(2))       - pad);
        z1 = max(1, floor(bb(3))       - pad);
        x2 = min(W, ceil(bb(1)+bb(4)) + pad);
        y2 = min(H, ceil(bb(2)+bb(5)) + pad);
        z2 = min(D, ceil(bb(3)+bb(6)) + pad);

        crop   = comp(y1:y2, x1:x2, z1:z2);
        thick  = imdilate(crop, se);
        smooth = imgaussfilt3(single(thick), options.GaussSigma);
        vmax   = max(smooth(:));
        if vmax < eps;  n_skipped = n_skipped+1;  continue;  end
        smooth = smooth / vmax;
        mask   = smooth > options.GaussThresh;
        if ~any(mask(:));  n_skipped = n_skipped+1;  continue;  end

        s = bwskel(mask, 'MinBranchLength', round(options.MinBranchLen));
        if ~any(s(:));  n_skipped = n_skipped+1;  continue;  end

        skel_clean(y1:y2, x1:x2, z1:z2) = skel_clean(y1:y2, x1:x2, z1:z2) | s;
        n_kept = n_kept + 1;

        if options.Verbose && mod(k, max(1,floor(CC.NumObjects/10))) == 0
            fprintf('  %d / %d  (%.0fs)\n', k, CC.NumObjects, toc(t0))
        end
    end

    if options.Verbose
        fprintf('Step 1 done: kept %d, skipped %d, voxels %d\n', ...
                n_kept, n_skipped, sum(skel_clean(:)))
    end

    % ── Step 2: remove chord fingers (crossing fingers with 2 contact points) ─
    if options.RemoveChords
        skel_clean = removeChords(skel_clean, kernel26, ...
                                  options.MaxChordLength, options.Verbose);
    end

    if options.Verbose
        fprintf('Final voxels: %d\n', sum(skel_clean(:)))
    end
end


% ═════════════════════════════════════════════════════════════════════════════
%  Local function: removeChords
% ═════════════════════════════════════════════════════════════════════════════

function skel = removeChords(skel, kernel26, maxLen, verbose)
% Detect and remove segments that bridge two points on the same loop.
%
% Strategy
% --------
% 1. Find branch points (≥3 neighbours).
% 2. Remove branch points → remaining voxels form "edge segments".
% 3. For each edge segment:
%      a. Does it have a free endpoint? → finger (handled by MinBranchLen), skip.
%      b. Does it connect to exactly 2 branch-point clusters? → potential chord.
%      c. Is it shorter than MaxChordLength?               → probable chord.
%      d. Are the 2 BP clusters still connected after removing this segment?
%         (i.e. the segment is NOT a bridge in the graph)  → confirmed chord.
%      If all of b, c, d: remove the segment.
% 4. Re-run endpoint pruning to clean dangling stubs left at former BP sites.

    n_removed = 0;
    max_passes = 5;   % iterate in case removal creates new chords

    for pass = 1:max_passes

        % ── branch points ─────────────────────────────────────────────────────
        nc       = round(convn(single(skel), kernel26, 'same'));
        bp_mask  = skel & (nc >= 3);
        if ~any(bp_mask(:))
            break
        end

        % label BP clusters (adjacent BPs form one logical node)
        bp_lab   = bwlabeln(bp_mask, 26);

        % segments = skeleton minus branch points
        segs     = skel & ~bp_mask;
        CC_segs  = bwconncomp(segs, 26);

        % neighbour count within segs only (to detect free endpoints)
        nc_segs  = round(convn(single(segs), kernel26, 'same'));

        % dilated BP mask for adjacency queries
        bp_dil   = imdilate(bp_mask, ones(3,3,3,'logical'));

        removed_this_pass = false;

        for k = 1 : CC_segs.NumObjects
            idx   = CC_segs.PixelIdxList{k};
            n_vox = numel(idx);

            % ── (a) skip if has a free endpoint ───────────────────────────────
            if any(nc_segs(idx) <= 1)
                continue
            end

            % ── (b) find which BP clusters this segment touches ───────────────
            seg_mask = false(size(skel));
            seg_mask(idx) = true;
            seg_dil  = imdilate(seg_mask, ones(3,3,3,'logical'));
            adj_bp_labels = unique(bp_lab(seg_dil & bp_mask));
            adj_bp_labels(adj_bp_labels == 0) = [];

            if numel(adj_bp_labels) ~= 2
                continue   % not a simple 2-endpoint bridge
            end

            % ── (c) length check ──────────────────────────────────────────────
            if n_vox > maxLen
                continue   % too long to be a crossing finger
            end

            % ── (d) connectivity check: are the 2 BP clusters still connected  ─
            %    after removing this segment?
            %    We test by temporarily removing the segment and checking if
            %    a voxel from BP cluster 1 can reach BP cluster 2.
            test_skel = skel & ~seg_mask;

            % pick one seed voxel from each BP cluster
            seed1_idx = find(bp_lab == adj_bp_labels(1), 1);
            seed2_idx = find(bp_lab == adj_bp_labels(2), 1);

            % flood-fill from seed1, check if seed2 is reached
            seed1 = false(size(skel));  seed1(seed1_idx) = true;
            filled = imreconstruct(seed1, test_skel);   % morphological flood-fill
            still_connected = filled(seed2_idx);

            if ~still_connected
                continue   % this segment is a bridge — essential, keep it
            end

            % ── confirmed chord: remove it ────────────────────────────────────
            skel(idx) = false;
            n_removed = n_removed + 1;
            removed_this_pass = true;
        end

        % ── clean dangling stubs left at former branch-point sites ────────────
        skel = pruneEndpoints(skel, kernel26);

        if ~removed_this_pass
            break
        end
    end

    if verbose
        fprintf('Chord removal: %d chords removed\n', n_removed)
    end
end


% ═════════════════════════════════════════════════════════════════════════════
%  Local function: pruneEndpoints
% ═════════════════════════════════════════════════════════════════════════════

function skel = pruneEndpoints(skel, kernel26)
% Iteratively remove voxels with exactly 1 neighbour (free endpoints).
    while true
        nc        = convn(single(skel), kernel26, 'same');
        endpoints = skel & (nc <= 1);
        if ~any(endpoints(:))
            break
        end
        skel(endpoints) = false;
    end
end