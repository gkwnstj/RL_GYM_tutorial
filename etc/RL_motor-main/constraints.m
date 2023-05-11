clc
max_tq = max(TqRefSat);

max_tq

max_tq_index = find(TqRefSat == max_tq);

max_tq_index


id_sens_when_max_tq = id_sens(max_tq_index);

id_sens_when_max_tq

iq_sens_when_max_tq = iq_sens(max_tq_index);

iq_sens_when_max_tq

Ismax = sqrt((id_sens_when_max_tq)^2 + (iq_sens_when_max_tq)^2);

Ismax % 366.3094



vd_ref_when_max_tq = vd_ref(max_tq_index);

vd_ref_when_max_tq

vq_ref_when_max_tq = vq_ref(max_tq_index);

vq_ref_when_max_tq

Vsmax = sqrt(vd_ref_when_max_tq^2 + vq_ref_when_max_tq^2);

Vsmax

%%
max_Vd = max(vd_ref)
max_Vq = max(vq_ref)

max_Vd_index = find(vq_ref == max_Vd);

max_Vd_index

vd_ref(max_Vd_index)
vq_ref(max_Vd_index)


%%
max_Vq_index = find(vq_ref == max_Vq);

max_Vq_index

vd_ref(max_Vq_index)
vq_ref(max_Vq_index)