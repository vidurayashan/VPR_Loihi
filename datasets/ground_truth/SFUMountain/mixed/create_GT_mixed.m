
GT.GThard = sparse(logical(repmat(eye(385), 6, 6)));
GT.GTsoft = sparse(imdilate(full(GT.GThard), strel("disk", 6)));
GT.version = 0;

GT.Info.GThard_source = 'repmat(eye(385), 6, 6)';
GT.Info.GTsoft_cmd = 'imdilate(GT.GThard, strel("disk", 6))';
GT.Info.skip_rate = 1;
GT.Info.conditions = 'sequence: dry-dusk-jan-nov-sept-wet';

save('gt.mat', 'GT');