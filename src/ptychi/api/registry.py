# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/pty-chi/blob/main/LICENSE

import ptychi.api.enums as enums
import ptychi.api.options.ad_general as ad_general
import ptychi.api.options.ad_ptychography as ad_ptychography
import ptychi.api.options.base as base
import ptychi.api.options.bh as bh
import ptychi.api.options.dm as dm
import ptychi.api.options.lsqml as lsqml
import ptychi.api.options.pie as pie
import ptychi.api.options.raar as raar


__all__ = ["RECONSTRUCTOR_OPTIONS_MAP"]


RECONSTRUCTOR_OPTIONS_MAP = {
    enums.Reconstructors.base: {
        "object_options": base.ObjectOptions,
        "probe_options": base.ProbeOptions,
        "probe_position_options": base.ProbePositionOptions,
        "opr_mode_weight_options": base.OPRModeWeightsOptions,
        "reconstructor_options": base.ReconstructorOptions,
    },
    enums.Reconstructors.AD_GENERAL: {
        "object_options": base.ObjectOptions,
        "probe_options": base.ProbeOptions,
        "probe_position_options": base.ProbePositionOptions,
        "opr_mode_weight_options": base.OPRModeWeightsOptions,
        "reconstructor_options": ad_general.AutodiffReconstructorOptions,
    },
    enums.Reconstructors.AD_PTYCHO: {
        "object_options": ad_ptychography.AutodiffPtychographyObjectOptions,
        "probe_options": ad_ptychography.AutodiffPtychographyProbeOptions,
        "probe_position_options": ad_ptychography.AutodiffPtychographyProbePositionOptions,
        "opr_mode_weight_options": ad_ptychography.AutodiffPtychographyOPRModeWeightsOptions,
        "reconstructor_options": ad_ptychography.AutodiffPtychographyReconstructorOptions,
    },
    enums.Reconstructors.LSQML: {
        "object_options": lsqml.LSQMLObjectOptions,
        "probe_options": lsqml.LSQMLProbeOptions,
        "probe_position_options": lsqml.LSQMLProbePositionOptions,
        "opr_mode_weight_options": lsqml.LSQMLOPRModeWeightsOptions,
        "reconstructor_options": lsqml.LSQMLReconstructorOptions,
    },
    enums.Reconstructors.PIE: {
        "object_options": pie.PIEObjectOptions,
        "probe_options": pie.PIEProbeOptions,
        "probe_position_options": pie.PIEProbePositionOptions,
        "opr_mode_weight_options": pie.PIEOPRModeWeightsOptions,
        "reconstructor_options": pie.PIEReconstructorOptions,
    },
    enums.Reconstructors.EPIE: {
        "object_options": pie.PIEObjectOptions,
        "probe_options": pie.PIEProbeOptions,
        "probe_position_options": pie.PIEProbePositionOptions,
        "opr_mode_weight_options": pie.PIEOPRModeWeightsOptions,
        "reconstructor_options": pie.EPIEReconstructorOptions,
    },
    enums.Reconstructors.RPIE: {
        "object_options": pie.PIEObjectOptions,
        "probe_options": pie.PIEProbeOptions,
        "probe_position_options": pie.PIEProbePositionOptions,
        "opr_mode_weight_options": pie.PIEOPRModeWeightsOptions,
        "reconstructor_options": pie.RPIEReconstructorOptions,
    },
    enums.Reconstructors.DM: {
        "object_options": dm.DMObjectOptions,
        "probe_options": dm.DMProbeOptions,
        "probe_position_options": dm.DMProbePositionOptions,
        "opr_mode_weight_options": dm.DMOPRModeWeightsOptions,
        "reconstructor_options": dm.DMReconstructorOptions,
    },
    enums.Reconstructors.RAAR: {
        "object_options": raar.RAARObjectOptions,
        "probe_options": raar.RAARProbeOptions,
        "probe_position_options": raar.RAARProbePositionOptions,
        "opr_mode_weight_options": raar.RAAROPRModeWeightsOptions,
        "reconstructor_options": raar.RAARReconstructorOptions,
    },
    enums.Reconstructors.BH: {
        "object_options": bh.BHObjectOptions,
        "probe_options": bh.BHProbeOptions,
        "probe_position_options": bh.BHProbePositionOptions,
        "opr_mode_weight_options": bh.BHOPRModeWeightsOptions,
        "reconstructor_options": bh.BHReconstructorOptions,
    },
}
