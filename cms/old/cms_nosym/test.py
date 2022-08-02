import subprocess


cmd_lp4 = ["../../dptc cms.dpt ip_harness.p4 linker_config.json cms_build"]
#ret = subprocess.run(cmd_lp4,shell=True)


cmd_tof = ["cd cms_build; make build"]
ret = subprocess.run(cmd_tof,shell=True)
#print(ret.returncode)

cmd_tst = ["cd cms_build/lucid/pipe; wc -l lucid.bfa"]
ret = subprocess.run(cmd_tst, shell=True)

'''
with open("lucid/pipe/lucid.bfa",'r') as f:
    lines = f.readlines()
    if len(lines) < 1:
        print("failed")

'''
