#pragma once
#include "common.h"

struct State
{
    int location;
    int timestep;
    int orientation;
    bool is_tasking_wait;

    State wait() const {return State(location, timestep + 1, orientation); }

    struct Hasher
    {
        std::size_t operator()(const State& n) const
        {
            size_t loc_hash = std::hash<int>()(n.location);
            size_t time_hash = std::hash<int>()(n.timestep);
            size_t ori_hash = std::hash<int>()(n.orientation);
            return (time_hash ^ (loc_hash << 1) ^ (ori_hash << 2));
        }
    };

    void operator = (const State& other)
    {
        timestep = other.timestep;
        location = other.location;
        orientation = other.orientation;
        is_tasking_wait = other.is_tasking_wait;
    }

    bool operator == (const State& other) const
    {
        return timestep == other.timestep &&
               location == other.location &&
               orientation == other.orientation &&
               is_tasking_wait == other.is_tasking_wait;
    }

    bool operator != (const State& other) const
    {
        return timestep != other.timestep ||
               location != other.location ||
               orientation != other.orientation ||
               is_tasking_wait != other.is_tasking_wait;
    }

    State(): location(-1), timestep(-1), orientation(-1), is_tasking_wait(false) {}
    // State(int loc): loc(loc), timestep(0), orientation(0) {}
    // State(int loc, int timestep): loc(loc), timestep(timestep), orientation(0) {}
    // State(
    //     int location,
    //     int timestep = -1,
    //     int orientation = -1): location(location),
    //                            timestep(timestep),
    //                            orientation(orientation) {}
    State(
        int location,
        int timestep = -1,
        int orientation = -1,
        bool is_tasking_wait = false): location(location),
                                       timestep(timestep),
                                       orientation(orientation),
                                       is_tasking_wait(is_tasking_wait) {}
    State(
        const State& other) {
            location = other.location;
            timestep = other.timestep;
            orientation = other.orientation;
            is_tasking_wait = other.is_tasking_wait;
        }
};

std::ostream & operator << (std::ostream &out, const State &s);


typedef std::vector<State> Path;

std::ostream & operator << (std::ostream &out, const Path &path);