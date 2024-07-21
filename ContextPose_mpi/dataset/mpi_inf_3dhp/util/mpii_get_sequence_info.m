function [bg_augmentable, ub_augmentable, lb_augmentable, chair_augmentable, fps, num_frames] = mpii_get_sequence_info(subject_id, sequence)
ub_augmentable = false;
lb_augmentable = false;
bg_augmentable = false;
chair_augmentable = false;
fps = 25;
switch subject_id
    case 1
        switch sequence
            case 1       
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6416;
            case 2
                ub_augmentable = true; %The LB masks are bad, so skip putting textures there and in the BG
                chair_augmentable = true;
				num_frames = 12430;
				fps = 50;
            otherwise
        end
    case 2
        switch sequence
            case 1
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6502;
            case 2
                bg_augmentable = true;
                chair_augmentable = true;
                ub_augmentable = true;
                lb_augmentable = true;
				num_frames = 6081;
        end        
    case 3
        switch sequence
			fps = 50;
            case 1
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 12488;
            case 2
                bg_augmentable = true;
                chair_augmentable = true;
                ub_augmentable = true;
                lb_augmentable = true;
				num_frames = 12283;
        end 
    case 4
        switch sequence
            case 1       
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6171;
            case 2
                chair_augmentable = true; %The LB masks are bad, so skip putting textures there and in the BG
                ub_augmentable = true;
				num_frames = 6675;
        end
    case 5
        switch sequence
			fps = 50;
            case 1       
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 12820;
            case 2
                chair_augmentable = true;
                ub_augmentable = true;
                bg_augmentable = true;
                lb_augmentable = true;
				num_frames = 12312;
            otherwise
        end
    case 6
        switch sequence
            case 1       
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6188;
            case 2
                ub_augmentable = true;
                lb_augmentable = true;
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6145;
            otherwise
        end
    case 7
        switch sequence
            case 1
                bg_augmentable = true;
                chair_augmentable = true;
                ub_augmentable = true;
                lb_augmentable = true;
				num_frames = 6239;
            case 2
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6320;
        end        
    case 8
        switch sequence
            case 1       
                bg_augmentable = true;
                chair_augmentable = true;
                ub_augmentable = true;
                lb_augmentable = true;
				num_frames = 6468;
            case 2
                bg_augmentable = true;
                chair_augmentable = true;
				num_frames = 6054;
        end        
end
end
