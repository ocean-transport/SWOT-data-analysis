import os
import xarray as xr
import tatsu_swot_utils as tatsu_swot


def remap_quality_flags(swath):
    """
    A simple script to remap quality flags >:(
    """

    if not "quality_flag" in swath:
        return

    flags = swath.quality_flag
    flags.values[flags.values==5.] = 1
    flags.values[flags.values==10.] = 2
    flags.values[flags.values==20.] = 3
    flags.values[flags.values==30.] = 4
    flags.values[flags.values==50.] = 5
    flags.values[flags.values==70.] = 6
    flags.values[flags.values==100.] = 7
    flags.values[flags.values==101.] = 8
    flags.values[flags.values==102.] = 9

    swath.quality_flag.values = flags.values
    
    return swath



def load_cycle(path,cycle="002",pass_ids=None,fields=None,subset=False,lats=[-90,90]):
    """
    A simple script to load locally saved cycles and passes

    Variables
    ---------
    path: string
    cycle: string
    pass_ids:
    fields: iterable or None
    subset: Boolean
    lats: iterable

    Output
    ------

    Dependencies:
    ------------

    """
    # First check that the actual cycle is loaded..
    if not os.path.exists(f"{path}/cycle_{cycle}"):
        print(f"Can't find path {path}/cycle_{cycle}")
        # Return empty array
        return []
    
    # If no pass ID is specified just load them all
    if pass_ids==None:
        # List the passes we have for the cycle
        swot_passes = [f for f in os.listdir(f"{path}/cycle_{cycle}") if ".nc" in f]
    else:
        # Else look for each specific pass you want
        swot_passes = []
        for pass_id in pass_ids:
            passes = [f for f in os.listdir(f"{path}/cycle_{cycle}") if f"Unsmoothed_{cycle}_{pass_id}" in f]
            swot_passes = swot_passes + passes
    
    # sort passes by pass ID
    # Example swath file: SWOT_L3_LR_SSH_Unsmoothed_001_578_20230810T203149_20230810T210930_v1.0.2_agulhas.nc
    swot_passes = sorted(swot_passes, key=lambda x: int(x.split("_")[6]))

    passes = []
    for swot_pass in swot_passes:
        print(f"Loading {swot_pass}")
        try:
            # Open subsetted swaths
            swath = xr.open_dataset(f"{path}/cycle_{cycle}/{swot_pass}")
            # Load specific fields if you specified them
            if fields == None:
                fields = list(swath.variables)
            swath = swath[fields]
            # If you want to subset do so for each swath here
            if subset:
                swath = tatsu_swot.subset(swath,lats)
            # Add the cycle and pass as attributes
            swath = swath.assign_attrs(cycle=f"{cycle}",pass_ID=f"{swot_pass.split("_")[6]}")
            # Remap the quality flags to 0-10 values for discrete plotting
            if "quality_flag" in fields:
                swath = remap_quality_flags(swath)

            
            # Aggregate swaths
            passes.append(swath)
        except Exception as e:
            print("Whoops can't open dataset")
            print("An error occured:",e)

    return passes

