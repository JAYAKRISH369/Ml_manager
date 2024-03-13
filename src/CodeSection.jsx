import "./CodeSection.css";
function CodeSection(props){
  return(
    <div className="Section" >
<div className="Title">
{` ${props.name}`}
</div>
<pre className="Code" >
  {props.code}
</pre>
    </div>
  );
}
export default CodeSection;