import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sola.aux.other import round_to_sf

def plot_solution(domain, least_norm_property, resolving_kernels, enquiry_points, targets, 
                  true_property: None, upper_bound: None, lower_bound: None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, subplot_titles=('Property', 'Targets and Approximations'))
    no_of_traces = 5 # by default

    least_norm_property = least_norm_property.reshape(least_norm_property.shape[0],)
    upper_bound = upper_bound.reshape(upper_bound.shape[0],)
    lower_bound = lower_bound.reshape(lower_bound.shape[0],)
    
    for step in range(len(resolving_kernels)):
        # LSQR Property
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color='#FFA500', width=4),
                name='Property',
                x=enquiry_points,
                y=least_norm_property
            ),
            row=1, col=1
        )
        # Avg kernels
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color='#DF0000', width=4),
                name='Avg kernel: ' + str(round_to_sf(enquiry_points[step], 2)),
                x=domain,
                y=resolving_kernels[step, :]
            ),
            row=2, col=1
        )
        # Targets
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color='#000000', width=4),
                name='Target: ' + str(round_to_sf(enquiry_points[step], 2)),
                x=domain,
                y=targets[step, :]
            ),
            row=2, col=1
        )

        # Add a vertical line to the first subplot
        fig.add_trace(go.Scatter(
            visible=False,
            line=dict(color='red', width=2, dash='dash'),  # Customize color and style of the line
            x=[enquiry_points[step], enquiry_points[step]],    # Specify x-coordinates of the line
            y=[lower_bound[step], upper_bound[step]],            # Specify y-coordinates of the line
        ), 
        row=1, col=1)

        # Add a vertical line to the second subplot
        fig.add_trace(go.Scatter(
            visible=False,
            line=dict(color='red', width=2, dash='dash'),  # Customize color and style of the line
            x=[enquiry_points[step], enquiry_points[step]],    # Specify x-coordinates of the line
            y=[min(resolving_kernels[step, :]), max(resolving_kernels[step, :])],            # Specify y-coordinates of the line
        ), 
        row=2, col=1)
        # True property
        if true_property is not None:
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='#32EA26', width=4),
                    name='True property',
                    x=enquiry_points,
                    y=true_property
                ),
                row=1, col=1
            )
            if step == 0: 
                no_of_traces += 1
        # Upper bound
        if upper_bound is not None:
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='#00CED1', width=4),
                    name='Upper bound',
                    x=enquiry_points,
                    y=upper_bound
                ),
                row=1, col=1
            )
            if step == 0: 
                no_of_traces += 1
        # Lower bound
        if lower_bound is not None:
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color='#00CED1', width=4),
                    name='Lower Bound',
                    x=enquiry_points,
                    y=lower_bound
                ),
                row=1, col=1
            )
            if step == 0: 
                no_of_traces += 1

    fig.update_xaxes(title_text="Domain", row=2, col=1)

    fig.data[0].visible = True
    fig.data[1].visible = True

    steps = []

    for i in range(len(resolving_kernels)):
        step = dict(
            method="update",
            args=[{"visible": [False] * (no_of_traces*len(resolving_kernels) + 1)},
                  {"title": "Slider"}],
        )
        for j in range(no_of_traces):
            step["args"][0]["visible"][no_of_traces*i + j] = True # Make true solution visible

        steps.append(step)
        
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="Subplots with Slider"
    )

    fig.show()