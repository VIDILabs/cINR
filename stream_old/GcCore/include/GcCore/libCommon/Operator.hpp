#pragma once

    //---------------------------------------------------------------------------------------------
    inline bool operator < (const uint4 &left, const uint4 &right)
    {
        if(left.x < right.x) return true;
        else if(left.x > right.x) return false;
        
        //x are equals
        if(left.y < right.y) return true;
        else if(left.y > right.y) return false;

        //x && y are equals
        if(left.z < right.z) return true;
        else if(left.z > right.z) return false;

        //x && y && z are equals
        if(left.w < right.w) return true;
        else if(left.w > right.w) return false;

        //x && y && z && w are equals
        return false;
    }