function [localFeaturesTemp, localFeaturesIdx] = checkHandRadius(localFeatures, skeletonFeatures, radius)

localFeaturesTemp = [];
localFeaturesIdx = [];
for k = 1:size(localFeatures, 1)
   
    frameNum = localFeatures(k, 1);
    fScale = localFeatures(k, 37);
    
    trajStartX = localFeatures(k, 2); %* fScale;
    trajStartY = localFeatures(k, 3); % * fScale;
    trajEndX = localFeatures(k, 30); % * fScale;
    trajEndY = localFeatures(k, 31); % * fScale;
    trajStdX = localFeatures(k, 34);
    trajStdY = localFeatures(k, 35);
    
    % left hand
    handLeftStartX = skeletonFeatures.HandLeft(frameNum - 15 + 1, 8)/3;
    handLeftStartY = skeletonFeatures.HandLeft(frameNum - 15 + 1, 9)/3;
    handLeftEndX = skeletonFeatures.HandLeft(frameNum, 8)/3;
    handLeftEndY = skeletonFeatures.HandLeft(frameNum, 9)/3;
    
    % right hand
    handRightStartX = skeletonFeatures.HandRight(frameNum - 15 + 1, 8)/3;
    handRightStartY = skeletonFeatures.HandRight(frameNum - 15 + 1, 9)/3;  
    handRightEndX = skeletonFeatures.HandRight(frameNum, 8)/3;
    handRightEndY = skeletonFeatures.HandRight(frameNum, 9)/3;
    
    % left hand
    distLeftStart = pdist([trajStartX, trajStartY; handLeftStartX, handLeftStartY]);
    distLeftEnd = pdist([trajEndX, trajEndY; handLeftEndX, handLeftEndY]);
    distRightStart = pdist([trajStartX, trajStartY; handRightStartX, handRightStartY]);
    distRightEnd = pdist([trajEndX, trajEndY; handRightEndX, handRightEndY]);
%     if (handLeftStartX >= trajStartX - radius && handLeftStartX <= trajStartX + radius ...
%             && handLeftStartY >= trajStartY - radius && handLeftStartY <= trajStartY + radius ...
%             && handLeftEndX >= trajEndX - radius && handLeftEndX <= trajEndX + radius ...
%             && handLeftEndY >= trajEndY - radius && handLeftEndY <= trajEndY + radius)
% %             && trajStdX >= 1 && trajStdY >= 1)
    if distLeftStart <= radius && distLeftEnd <= radius
        localFeaturesTemp = [localFeaturesTemp; localFeatures(k, :)];
        localFeaturesIdx = [localFeaturesIdx; k];
    % right hand
%     elseif (handRightStartX >= trajStartX - radius && handRightStartX <= trajStartX + radius ...
%             && handRightStartY >= trajStartY - radius && handRightStartY <= trajStartY + radius ...
%             && handRightEndX >= trajEndX - radius && handRightEndX <= trajEndX + radius ...
%             && handRightEndY >= trajEndY - radius && handRightEndY <= trajEndY + radius)
% %             && trajStdX >= 1 && trajStdY >= 1)
    elseif distRightStart <= radius && distRightEnd <= radius
        localFeaturesTemp = [localFeaturesTemp; localFeatures(k, :)];
        localFeaturesIdx = [localFeaturesIdx; k];
    end
end

end
