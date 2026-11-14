"""Resume a reboot-interrupted frozen study, reusing verified completed folds."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dscnn_baseline.common import config,new_directory,source,write_json
from dscnn_baseline.data import development_identity
from dscnn_baseline.study import code_identity,choose_weight,run_child,summarize
from dscnn_baseline.weak_labels import require_file_label_approval


def inspect(args):
    previous=Path(args.resume_from).resolve()
    plan=source.read_json(previous/"execution_plan.json")
    cfg,_=config()
    identity=development_identity()
    supervision=require_file_label_approval(args.file_label_approval)
    if plan["config"]!=cfg or plan["code_identity"]!=code_identity() or plan["development_identity"]!=identity or plan["supervision"]!=supervision:
        raise ValueError("Frozen model code, configuration, dataset or supervision changed since interruption")
    if (previous/"frozen_model.json").exists() or (previous/"historical_holdout").exists():
        raise ValueError("This resumer only handles interruption during CV before holdout access")
    reused,interrupted={},[]
    for weight in (1,2,4):
        for fold in range(4):
            path=ROOT/"runs"/f"{plan['run_prefix']}_w{weight}_f{fold}"
            if not path.is_dir():continue
            status=source.read_json(path/"status.json")
            run=source.read_json(path/"run.json")
            if run["code_sha256"]!=plan["code_identity"] or run["config"]!=cfg or run["development_identity"]!=identity or run["supervision"]!=supervision:
                raise ValueError(f"Prior fold provenance changed: {path}")
            if status.get("exit_code")==0:
                if source.sha256(path/"classifier.keras")!=status["model_sha256"]:
                    raise ValueError("Prior completed checkpoint hash changed")
                history=source.read_csv(path/"history.csv")
                if len(history)!=status["epochs_completed"] or status["optimizer_steps"]!=len(history)*cfg["training"]["steps_per_epoch"]:
                    raise ValueError("Prior completed fold budget is inconsistent")
                reused[(weight,fold)]=path
            elif status.get("first_error") or (path/"first_error.txt").exists():
                raise ValueError(f"Recorded model failure must be reviewed before retrying: {path}")
            else:
                interrupted.append(str(path))
    return plan,supervision,reused,interrupted


def run(args):
    plan,supervision,reused,interrupted=inspect(args)
    if args.inspect_only:
        print(source.fingerprint(plan),"REUSE",{str(k):str(v) for k,v in reused.items()},"PRESERVE_INTERRUPTED",interrupted)
        return
    output=new_directory(args.output)
    status={"kind":"dscnn_real_audio_study","stage":"resume_validation","exit_code":None,
            "completed_folds":0,"completed_weights":[],"test_evaluations":0,
            "installation_executed":False,"cpu_training_fallback":False,"resumed_after_reboot":True,
            "reused_folds":len(reused),"new_training_executions":0,"pid":os.getpid()}
    write_json(output/"status.json",status)
    started=time.monotonic()
    try:
        for name in ("gpu_preflight","gpu_export_preflight","weak_label_gpu_preflight"):
            preflight=source.read_json(output.parent/name/"status.json")
            if preflight.get("passed") is not True or preflight.get("exit_code")!=0:
                raise ValueError(f"Required preflight did not pass: {name}")
        if source.read_json(output.parent/"weak_label_gpu_preflight/status.json")["weak_labels_code_sha256"]!=source.sha256(ROOT/"dscnn_baseline/weak_labels.py"):
            raise ValueError("File-label optimizer changed since GPU preflight")
        resumed_plan=dict(plan,run_prefix=args.run_prefix,resume_from=str(Path(args.resume_from).resolve()),
                          source_plan_sha256=source.sha256(Path(args.resume_from)/"execution_plan.json"),
                          reused_completed_folds={f"w{w}_f{f}":str(p) for (w,f),p in reused.items()},
                          interrupted_runs_preserved=interrupted,
                          resume_rule="Reuse only exit0 folds with identical code/config/data and checkpoint hashes. Incomplete folds restart with original fold seed in a fresh directory; no optimizer state was saved.")
        write_json(output/"execution_plan.json",resumed_plan)
        supervision_args=["--supervision","file_labels","--file-label-approval",str(args.file_label_approval.resolve())]
        summaries=[]
        for weight in (1,2,4):
            directories=[]
            for fold in range(4):
                if plan["code_identity"]!=code_identity() or plan["development_identity"]!=development_identity() or plan["config"]!=config()[0]:
                    raise ValueError("Frozen study inputs changed after resume")
                status.update(stage="cross_validation",weight=weight,fold=fold)
                write_json(output/"status.json",status)
                if (weight,fold) in reused:
                    path=reused[(weight,fold)]
                    print("VERIFIED_COMPLETED_FOLD_REUSED",weight,fold,path,flush=True)
                else:
                    name=f"{args.run_prefix}_w{weight}_f{fold}"
                    run_child([sys.executable,"-B","-m","dscnn_baseline","train",*supervision_args,
                               "--fold",str(fold),"--run-name",name,"--device","gpu",
                               "--domestic-weight",str(weight),"--wandb-mode","disabled","--skip-export"],
                              output/f"w{weight}_fold{fold}.log")
                    path=ROOT/"runs"/name
                    status["new_training_executions"]+=1
                directories.append(path)
                status["completed_folds"]+=1
                write_json(output/"status.json",status)
            summaries.append(summarize(weight,directories,supervision,output))
            status["completed_weights"].append(weight)
            write_json(output/"status.json",status)
        selected=choose_weight(summaries)
        selected_path=output/f"w{selected['domestic_weight']}_cv_summary.json"
        decision={"selected_weight":selected["domestic_weight"],"refit_epochs":selected["refit_epochs"],
                  "decision_threshold":selected["decision_policy"]["threshold"],"summary":str(selected_path),
                  "summary_sha256":source.sha256(selected_path),"test_seen_for_this_selection":False}
        write_json(output/"decision.json",decision)
        status["stage"]="fixed_epoch_refit_and_int8_export";write_json(output/"status.json",status)
        refit_name=f"{args.run_prefix}_refit_w{selected['domestic_weight']}"
        run_child([sys.executable,"-B","-m","dscnn_baseline","refit","--cv-summary",str(selected_path),
                   *supervision_args,"--run-name",refit_name,"--device","gpu","--wandb-mode","disabled"],output/"refit.log")
        status["new_training_executions"]+=1
        refit=ROOT/"runs"/refit_name
        frozen={"model_sha256":source.sha256(refit/"classifier.keras"),
                "int8_sha256":source.sha256(refit/"export/dscnn_int8.tflite"),
                "development_identity":plan["development_identity"],"decision_sha256":source.sha256(output/"decision.json"),
                "decision_policy":selected["decision_policy"],
                "prediction_policy":"All 1s windows mean softmax per file, equal-file mean per source, then argmax and fixed target confidence rejection",
                "test_note":"Previously evaluated for YAMNet; historical comparison only, no fresh-test claim"}
        write_json(output/"frozen_model.json",frozen)
        status["stage"]="historical_holdout_evaluation";write_json(output/"status.json",status)
        run_child([sys.executable,"-B","-m","dscnn_baseline.evaluation","--refit",str(refit),
                   "--frozen",str(output/"frozen_model.json"),"--output",str(output/"historical_holdout")],output/"historical_holdout.log")
        status.update(stage="complete",exit_code=0,refit=str(refit),decision=decision,test_evaluations=1,
                      model_development_and_training_complete=True,board_validated=False,deployment_ready=False)
        print("RESUMED_DSCNN_STUDY_COMPLETE",output,flush=True)
    except Exception as error:
        status.update(stage="stopped_on_first_failure",exit_code=1,first_error=f"{type(error).__name__}: {error}")
        (output/"first_error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"]=time.monotonic()-started
        write_json(output/"status.json",status)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-prefix",required=True)
    parser.add_argument("--file-label-approval",type=Path,required=True)
    parser.add_argument("--inspect-only",action="store_true")
    run(parser.parse_args())
