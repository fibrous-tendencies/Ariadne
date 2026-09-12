using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Theseus.Interop;

/// <summary>
/// Raw P/Invoke declarations for theseus.dll.
/// Every function here maps 1-to-1 to an <c>extern "C"</c> symbol
/// in <c>rust/src/ffi.rs</c>.  Prefer the managed <see cref="TheseusSolver"/>
/// wrapper for production use.
/// </summary>
internal static class TheseusInterop
{
    const string DLL = "theseus";
    static IntPtr TheseusModule = IntPtr.Zero;

    static TheseusInterop()
    {
        NativeLibrary.SetDllImportResolver(typeof(TheseusInterop).Assembly, ResolveTheseus);
    }

    static IntPtr ResolveTheseus(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        string? fileName = null;
        if (name == DLL)
            fileName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "theseus.dll" : "libtheseus.dylib";

        if (fileName == null)
            return IntPtr.Zero;

        string? dir = Path.GetDirectoryName(assembly.Location);
        if (string.IsNullOrEmpty(dir))
            return IntPtr.Zero;

        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && !RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            return IntPtr.Zero;

        string path = Path.Combine(dir, fileName);

        if (NativeLibrary.TryLoad(path, assembly, searchPath, out IntPtr handle))
        {
            TheseusModule = handle;
            return handle;
        }
        return IntPtr.Zero;
    }

    // ── Error reporting ──────────────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_last_error(byte[] buf, nuint buf_len);

    // ── Handle lifecycle ─────────────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr theseus_create(
        nuint num_edges, nuint num_nodes, nuint num_free,
        nuint[] coo_rows, nuint[] coo_cols, double[] coo_vals, nuint coo_nnz,
        nuint[] free_node_indices, nuint[] fixed_node_indices, nuint num_fixed,
        double[] loads, double[] fixed_positions,
        double[] q_init, double[] lower_bounds, double[] upper_bounds);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr theseus_create_with_variable_supports(
        nuint num_edges, nuint num_nodes, nuint num_free,
        nuint[] coo_rows, nuint[] coo_cols, double[] coo_vals, nuint coo_nnz,
        nuint[] free_node_indices, nuint[] fixed_node_indices, nuint num_fixed,
        double[] loads, double[] fixed_positions,
        double[] q_init, double[] lower_bounds, double[] upper_bounds,
        nuint num_variable_supports,
        nuint[] variable_node_indices,
        int[] support_kinds,
        double[] support_lambdas,
        double[] sphere_radii,
        byte[] roller_enabled,
        double[] roller_lower,
        double[] roller_upper,
        double[] rail_start,
        double[] rail_end,
        nuint[] nurbs_offsets,
        nuint[] nurbs_lengths,
        double[] nurbs_data,
        nuint nurbs_data_len);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern void theseus_free(IntPtr handle);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_cancel(IntPtr handle);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_begin_cancel_scope(
        IntPtr handle,
        ref ulong out_scope_id);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_cancel_scope(IntPtr handle, ulong scope_id);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_complete_cancel_scope(IntPtr handle, ulong scope_id);

    // ── Objective registration ───────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_target_xyz(
        IntPtr handle, double weight,
        nuint[] node_indices, nuint num_nodes,
        double[] target_xyz, int reduction);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_target_xy(
        IntPtr handle, double weight,
        nuint[] node_indices, nuint num_nodes,
        double[] target_xy, int reduction);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_target_plane(
        IntPtr handle, double weight,
        nuint[] node_indices, nuint num_nodes,
        double[] target_xyz,
        double[] origin, double[] x_axis, double[] y_axis, int reduction);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_planar_constraint_along_direction(
        IntPtr handle, double weight,
        nuint[] node_indices, nuint num_nodes,
        double[] origin, double[] x_axis, double[] y_axis, double[] direction);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_target_length(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double[] targets);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_target_force(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double[] targets);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_length_variation(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double sharpness,
        byte use_normalized_variance,
        int normalization_strategy);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_force_variation(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double sharpness,
        byte use_normalized_variance,
        int normalization_strategy);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_sum_force_length(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_min_length(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double[] thresholds, double sharpness);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_max_length(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double[] thresholds, double sharpness);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_min_force(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double[] thresholds, double sharpness);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_max_force(
        IntPtr handle, double weight,
        nuint[] edge_indices, nuint num_edges,
        double[] thresholds, double sharpness);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_rigid_set_compare(
        IntPtr handle, double weight,
        nuint[] node_indices, nuint num_nodes,
        double[] target_xyz);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_reaction_direction(
        IntPtr handle, double weight,
        nuint[] anchor_indices, nuint num_anchors,
        double[] target_dirs);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_reaction_direction_magnitude(
        IntPtr handle, double weight,
        nuint[] anchor_indices, nuint num_anchors,
        double[] target_dirs, double[] target_mags);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_reaction_magnitude(
        IntPtr handle, double weight,
        nuint[] anchor_indices, nuint num_anchors,
        double[] target_dirs, double[] target_mags,
        int behavior, int sign_semantics);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_add_reaction_direction_magnitude_with_options(
        IntPtr handle, double weight,
        nuint[] anchor_indices, nuint num_anchors,
        double[] target_dirs, double[] target_mags,
        int behavior, int sign_semantics);

    // ── Solver options ───────────────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_solver_options(
        IntPtr handle,
        nuint max_iterations, double abs_tol, double rel_tol,
        double barrier_weight, double barrier_sharpness,
        double anchor_saturation_lambda);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_q_parameterization_mode(
        IntPtr handle,
        int mode);

    // ── Self-weight configuration ────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_self_weight_prescribed(
        IntPtr handle,
        double[] linear_densities, double[] gravity,
        nuint max_iters, double tolerance, double relaxation);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_self_weight_sizing(
        IntPtr handle,
        double rho, double sigma, double[] gravity,
        nuint max_iters, double tolerance, double relaxation);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_clear_self_weight(IntPtr handle);

    // ── Pressure load configuration ──────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_pressure(
        IntPtr handle,
        nuint num_faces,
        nuint[] face_offsets, nuint[] face_vertices,
        double[] pressures,
        nuint max_iters, double tolerance, double relaxation);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_pressure_hydrostatic(
        IntPtr handle,
        nuint num_faces,
        nuint[] face_offsets, nuint[] face_vertices,
        double rho_fluid, double g_magnitude, double z_datum,
        double[] up_direction,
        nuint max_iters, double tolerance, double relaxation);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_pressure_directional(
        IntPtr handle,
        nuint num_faces,
        nuint[] face_offsets, nuint[] face_vertices,
        double[] pressures, double[] direction,
        nuint max_iters, double tolerance, double relaxation);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_clear_pressure(IntPtr handle);

    // ── Progress callback ────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate byte NativeProgressCallback(
        nuint majorIteration, double loss, IntPtr xyz, nuint numNodes,
        IntPtr q, nuint numEdges);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_set_progress_callback(
        IntPtr handle,
        NativeProgressCallback? callback,
        nuint frequency);

    // ── Optimisation ─────────────────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_optimize(
        IntPtr handle,
        double[] out_xyz, double[] out_lengths, double[] out_forces,
        double[] out_q, double[] out_reactions,
        ref nuint out_iterations, ref byte out_converged);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_optimize_scoped(
        IntPtr handle,
        ulong scope_id,
        double[] out_xyz, double[] out_lengths, double[] out_forces,
        double[] out_q, double[] out_reactions,
        ref nuint out_iterations, ref byte out_converged);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_get_termination_reason(
        IntPtr handle,
        byte[] buffer,
        nuint buffer_len);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint theseus_get_loss_trace_len(IntPtr handle);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern nuint theseus_get_loss_trace(
        IntPtr handle,
        double[] out_loss_trace,
        nuint out_len);

    // ── Forward solve ────────────────────────────────────────

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_solve_forward(
        IntPtr handle,
        double[] out_xyz, double[] out_lengths, double[] out_forces,
        double[] out_q, double[] out_reactions);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_solve_forward_scoped(
        IntPtr handle,
        ulong scope_id,
        double[] out_xyz, double[] out_lengths, double[] out_forces,
        double[] out_q, double[] out_reactions);

    // ── Inverse solvers (experimental) ────────────────────────

    /// <summary>
    /// particular_method: 0 = Gram, 1 = Augmented (Moore–Penrose / Tikhonov),
    /// 2 = Sparse QR, 3 = Clarabel.
    /// linear_algebra: 0 = Direct, 1 = Iterative.
    /// Empty signs / lower / upper (length 0) means unconstrained on that channel.
    /// </summary>
    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_solve_inverse_fdm(
        IntPtr handle,
        double[] target_free_xyz, double regularization,
        int use_l2, nuint max_l1_iter, int particular_method, int linear_algebra,
        int enforce_zero_rx, int enforce_zero_ry, int enforce_zero_rz, int solve_for_q,
        int[] signs, nuint n_signs,
        double[] lower, nuint n_lower,
        double[] upper, nuint n_upper,
        nuint max_iter, double tol,
        double[] out_q, double[] out_xyz, double[] out_lengths,
        double[] out_forces, double[] out_reactions,
        ref nuint out_iterations, ref byte out_converged);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_solve_inverse_fdm_metric(
        IntPtr handle,
        double[] target_free_xyz, double regularization,
        int use_l2, nuint max_l1_iter, int particular_method, int linear_algebra,
        int enforce_zero_rx, int enforce_zero_ry, int enforce_zero_rz, int solve_for_q,
        int[] signs, nuint n_signs,
        double[] lower, nuint n_lower,
        double[] upper, nuint n_upper,
        nuint max_iter, double tol,
        int metric, double[]? q_ref, nuint n_q_ref, nuint max_outer,
        double[] out_q, double[] out_xyz, double[] out_lengths,
        double[] out_forces, double[] out_reactions,
        ref nuint out_iterations, ref byte out_converged, ref double out_geom_error);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_solve_inverse_fdm_metric_cwls(
        IntPtr handle,
        double[] target_free_xyz, double regularization, double cwls_damping,
        int use_l2, nuint max_l1_iter, int particular_method, int linear_algebra,
        int enforce_zero_rx, int enforce_zero_ry, int enforce_zero_rz, int solve_for_q,
        int[] signs, nuint n_signs,
        double[] lower, nuint n_lower,
        double[] upper, nuint n_upper,
        nuint max_iter, double tol,
        int metric, double[]? q_ref, nuint n_q_ref, nuint max_outer,
        double[] out_q, double[] out_xyz, double[] out_lengths,
        double[] out_forces, double[] out_reactions,
        ref nuint out_iterations, ref byte out_converged, ref double out_geom_error);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_solve_inverse_fdm_metric_phases(
        IntPtr handle,
        double[] target_free_xyz, double regularization, double cwls_damping,
        int use_l2, nuint max_l1_iter, int particular_method, int linear_algebra,
        int enforce_zero_rx, int enforce_zero_ry, int enforce_zero_rz, int solve_for_q,
        int[] signs, nuint n_signs,
        double[] lower, nuint n_lower,
        double[] upper, nuint n_upper,
        nuint max_iter, double tol,
        int metric, double[]? q_ref, nuint n_q_ref,
        nuint max_frozen_outer, nuint max_outer,
        double[] out_q, double[] out_xyz, double[] out_lengths,
        double[] out_forces, double[] out_reactions,
        ref nuint out_iterations, ref byte out_converged, ref double out_geom_error);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_rigidity_report_sizes(
        IntPtr handle,
        double[] target_free_xyz,
        int method,
        int include_rigid_bodies,
        nuint max_modes,
        ref nuint out_rank,
        ref nuint out_self_stress_count,
        ref nuint out_mechanism_raw_count,
        ref nuint out_mechanism_count,
        ref nuint out_rigid_count,
        ref nuint out_particular_len,
        ref nuint out_residual_len,
        ref nuint out_self_stress_len,
        ref nuint out_mechanism_len,
        ref nuint out_rigid_len);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_rigidity_report_fill(
        IntPtr handle,
        double[] out_particular_t,
        nuint particular_len,
        double[] out_residual,
        nuint residual_len,
        double[] out_self_stress,
        nuint self_stress_len,
        double[] out_mechanisms,
        nuint mechanism_len,
        double[] out_rigid_bodies,
        nuint rigid_len,
        ref double out_residual_ratio);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_retract_member_lengths(
        IntPtr handle,
        double[] initial_free_xyz,
        double[] target_lengths,
        nuint max_iterations,
        double tolerance,
        double[] out_free_xyz,
        ref nuint out_iterations,
        ref byte out_converged,
        ref double out_max_length_error,
        ref double out_residual_norm);

    [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)]
    public static extern int theseus_classify_prestress(
        IntPtr handle,
        double[] target_free_xyz,
        double[] prestress_t,
        double[] mechanisms,
        nuint mechanism_count,
        double tolerance,
        double[] out_eigenvalues,
        int[] out_classes,
        double[] out_rotated_mechanisms);

    /// <summary>
    /// Requests solver cancellation when exported by the native library.
    /// </summary>
    public static void TryCancel(IntPtr handle)
    {
        if (handle == IntPtr.Zero || TheseusModule == IntPtr.Zero)
            return;

        if (!NativeLibrary.TryGetExport(TheseusModule, "theseus_cancel", out IntPtr symbol))
            return;

        var cancel = Marshal.GetDelegateForFunctionPointer<TheseusCancelDelegate>(symbol);
        cancel(handle);
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int TheseusCancelDelegate(IntPtr handle);
}
