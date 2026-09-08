"""Deterministic bbox matching for canonical [ymin, xmin, ymax, xmax] boxes."""


def iou(a, b):
    ay1,ax1,ay2,ax2=a; by1,bx1,by2,bx2=b
    inter=max(0,min(ay2,by2)-max(ay1,by1))*max(0,min(ax2,bx2)-max(ax1,bx1)); ua=(ay2-ay1)*(ax2-ax1)+(by2-by1)*(bx2-bx1)-inter
    return inter/ua if ua else 0.0
def match_detections(detections, truth, minimum_iou=0.20, ambiguity_margin=0.05):
    matches=[]; used=set(); ambiguous=[]; unmatched=[]
    for d in detections:
        cand=sorted(((iou(d.bbox,t["bbox"]),t) for t in truth if t["object_id"] not in used),reverse=True,key=lambda x:(x[0],x[1]["object_id"]))
        if not cand or cand[0][0]<minimum_iou: unmatched.append(d.entity_id); continue
        if len(cand)>1 and cand[0][0]-cand[1][0]<ambiguity_margin: ambiguous.append(d.entity_id); continue
        used.add(cand[0][1]["object_id"]); matches.append((d.entity_id,cand[0][1]["object_id"],cand[0][0]))
    return matches,unmatched,ambiguous
